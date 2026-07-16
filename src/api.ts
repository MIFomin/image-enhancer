export type TaskStatus = 'pending' | 'decoding' | 'analyzing' | 'processing' | 'encoding' | 'completed' | 'cancelled' | 'error';

export interface TaskProgress {
  taskId: string;
  status: TaskStatus;
  progress: number; // 0 to 100
}

export interface CorrectionParams {
  brightness: number;
  contrast: number;
  saturation: number;
}

export interface TaskResult {
  blob: Blob;
  params: CorrectionParams;
}

interface TaskState {
  status: TaskStatus;
  progress: number;
  blob?: Blob;
  params?: CorrectionParams;
  resolveResult?: (result: TaskResult) => void;
  rejectResult?: (err: Error) => void;
  resultPromise?: Promise<TaskResult>;
}

export class ImageEnhancer {
  private worker: Worker;
  private tasks: Map<string, TaskState> = new Map();
  private listeners: Map<string, Set<Function>> = new Map();
  private taskCounter = 0;

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    
    this.worker.onmessage = (e) => this.handleWorkerMessage(e.data);
    this.worker.onerror = (err) => {
      console.error('[Worker Error]', err);
    };
  }

  private handleWorkerMessage(msg: any) {
    const task = this.tasks.get(msg.taskId);
    if (!task) return;

    if (msg.type === 'progress') {
      task.status = msg.status;
      task.progress = msg.progress;
      this.emit('statusChange', { 
        taskId: msg.taskId, 
        status: msg.status, 
        progress: msg.progress 
      });
    } 
    else if (msg.type === 'result') {
      task.status = 'completed';
      task.progress = 100;
      task.blob = msg.blob;
      task.params = msg.params;
      task.resolveResult?.({ blob: msg.blob, params: msg.params });
      this.emit('statusChange', { 
        taskId: msg.taskId, 
        status: 'completed', 
        progress: 100 
      });
    } 
    else if (msg.type === 'error') {
      task.status = 'error';
      task.rejectResult?.(new Error(msg.message));
      this.emit('statusChange', { 
        taskId: msg.taskId, 
        status: 'error', 
        progress: 0 
      });
    }
  }

  async submitTask(image: Blob): Promise<string> {
    const taskId = `task_${++this.taskCounter}_${Date.now()}`;
    
    const taskState: TaskState = {
      status: 'pending',
      progress: 0,
    };

    // Создаём Promise для ожидания результата
    const resultPromise = new Promise<TaskResult>((resolve, reject) => {
      taskState.resolveResult = resolve;
      taskState.rejectResult = reject;
    });
    
    taskState.resultPromise = resultPromise;
    this.tasks.set(taskId, taskState);

    // Отправляем задачу в воркер
    this.worker.postMessage({ type: 'start', taskId, image });

    this.emit('statusChange', { taskId, status: 'pending', progress: 0 });
    return taskId;
  }

  async getTaskStatus(taskId: string): Promise<TaskProgress> {
    const task = this.tasks.get(taskId);
    if (!task) throw new Error(`Task ${taskId} not found`);
    return { taskId, status: task.status, progress: task.progress };
  }

  async cancelTask(taskId: string): Promise<boolean> {
    const task = this.tasks.get(taskId);
    if (!task) return false;
    
    this.worker.postMessage({ type: 'cancel', taskId });
    task.status = 'cancelled';
    task.rejectResult?.(new Error('Task cancelled'));
    return true;
  }

  async getResult(taskId: string): Promise<TaskResult> {
    const task = this.tasks.get(taskId);
    if (!task) throw new Error(`Task ${taskId} not found`);
    
    // Если результат уже получен — возвращаем сразу
    if (task.blob && task.params) {
      return { blob: task.blob, params: task.params };
    }
    
    // Если задача завершилась с ошибкой
    if (task.status === 'error') throw new Error('Task failed');
    if (task.status === 'cancelled') throw new Error('Task cancelled');
    
    // Иначе ждём завершения
    return task.resultPromise!;
  }

  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: Function): void {
    this.listeners.get(event)?.delete(callback);
  }

  private emit(event: string, data: TaskProgress) {
    this.listeners.get(event)?.forEach(cb => cb(data));
  }

  destroy() {
    this.worker.terminate();
  }
}