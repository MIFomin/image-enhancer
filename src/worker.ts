import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
ort.env.wasm.numThreads = 1;

const vertexShaderSource = `#version 300 es
in vec2 a_position;
in vec2 a_texCoord;
out vec2 v_texCoord;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_texCoord = a_texCoord;
}`;

const fragmentShaderSource = `#version 300 es
precision mediump float;

in vec2 v_texCoord;
uniform sampler2D u_image;
uniform float u_brightness;
uniform float u_contrast;
uniform float u_saturation;

out vec4 outColor;

void main() {
  vec4 color = texture(u_image, v_texCoord);
  color.rgb += u_brightness;
  color.rgb = (color.rgb - 0.5) * u_contrast + 0.5;
  float luminance = dot(color.rgb, vec3(0.299, 0.587, 0.114));
  color.rgb = mix(vec3(luminance), color.rgb, u_saturation);
  outColor = clamp(color, 0.0, 1.0);
}`;

let currentTaskId: string | null = null;
let isCancelled = false;
let mlSession: ort.InferenceSession | null = null;

function log(taskId: string, message: string) {
  console.log(`[Worker ${taskId}] ${message}`);
}

function post(msg: any) {
  self.postMessage(msg);
}

function checkCancel() {
  if (isCancelled) throw new Error('TASK_CANCELLED');
}

async function initORT() {
  if (mlSession) return;
  
  console.log('[ONNX] Загрузка модели...');
  
  const modelPath = (import.meta as any).env?.BASE_URL 
    ? `${(import.meta as any).env.BASE_URL}model/model.onnx`
    : '/model/model.onnx';
  
  mlSession = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['wasm']
  });
  console.log('[ONNX] Модель загружена');
}

async function analyzeImageWithML(bitmap: ImageBitmap, taskId: string) {
  log(taskId, 'ONNX анализ изображения');
  
  await initORT();
  
  const analysisCanvas = new OffscreenCanvas(64, 64);
  const ctx = analysisCanvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0, 64, 64);
  const imageData = ctx.getImageData(0, 0, 64, 64);
  
  const pixels = new Float32Array(64 * 64 * 3);
  for (let i = 0, j = 0; i < imageData.data.length; i += 4, j += 3) {
    pixels[j] = imageData.data[i] / 255;
    pixels[j + 1] = imageData.data[i + 1] / 255;
    pixels[j + 2] = imageData.data[i + 2] / 255;
  }
  
  const inputTensor = new ort.Tensor('float32', pixels, [1, 64, 64, 3]);
  const feeds = { [mlSession!.inputNames[0]]: inputTensor };
  const results = await mlSession!.run(feeds);
  const output = results[mlSession!.outputNames[0]].data as Float32Array;
  
  return {
    brightness: output[0],
    contrast: output[1],
    saturation: output[2]
  };
}

async function processWithWebGL(bitmap: ImageBitmap, params: any, taskId: string, onProgress: (p: number) => void) {
  log(taskId, `Создание OffscreenCanvas ${bitmap.width}x${bitmap.height}`);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  
  const gl = canvas.getContext('webgl2');
  if (!gl) throw new Error('WebGL2 не поддерживается');

  const vs = gl.createShader(gl.VERTEX_SHADER)!;
  gl.shaderSource(vs, vertexShaderSource);
  gl.compileShader(vs);
  if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
    throw new Error('VERTEX_SHADER: ' + gl.getShaderInfoLog(vs));
  }

  const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
  gl.shaderSource(fs, fragmentShaderSource);
  gl.compileShader(fs);
  if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
    throw new Error('FRAGMENT_SHADER: ' + gl.getShaderInfoLog(fs));
  }

  const program = gl.createProgram()!;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error('Program: ' + gl.getProgramInfoLog(program));
  }
  gl.useProgram(program);

  const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const texCoords = new Float32Array([0, 1, 1, 1, 0, 0, 1, 0]);
  
  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  const posLoc = gl.getAttribLocation(program, 'a_position');
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  const texBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, texBuf);
  gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
  const texLoc = gl.getAttribLocation(program, 'a_texCoord');
  gl.enableVertexAttribArray(texLoc);
  gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

  gl.uniform1f(gl.getUniformLocation(program, 'u_brightness'), params.brightness);
  gl.uniform1f(gl.getUniformLocation(program, 'u_contrast'), params.contrast);
  gl.uniform1f(gl.getUniformLocation(program, 'u_saturation'), params.saturation);

  onProgress(60);
  checkCancel();

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  
  onProgress(80);
  checkCancel();

  gl.deleteTexture(texture);
  gl.deleteProgram(program);
  gl.deleteBuffer(posBuf);
  gl.deleteBuffer(texBuf);
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  return canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 });
}

async function processTask(taskId: string, image: Blob) {
  try {
    log(taskId, 'Начало обработки');
    
    post({ type: 'progress', taskId, status: 'decoding', progress: 5 });
    const bitmap = await createImageBitmap(image);
    log(taskId, `Декодировано: ${bitmap.width}x${bitmap.height}`);
    post({ type: 'progress', taskId, status: 'decoding', progress: 20 });
    checkCancel();

    post({ type: 'progress', taskId, status: 'analyzing', progress: 25 });
    const params = await analyzeImageWithML(bitmap, taskId);
    log(taskId, `ML параметры: ${JSON.stringify(params)}`);
    post({ type: 'progress', taskId, status: 'analyzing', progress: 40 });
    checkCancel();

    post({ type: 'progress', taskId, status: 'processing', progress: 45 });
    const resultBlob = await processWithWebGL(bitmap, params, taskId, (p) => {
      post({ type: 'progress', taskId, status: 'processing', progress: p });
    });
    log(taskId, `Результат: ${(resultBlob.size / 1024).toFixed(2)} KB`);
    checkCancel();

    post({ type: 'progress', taskId, status: 'encoding', progress: 90 });
    bitmap.close();

    post({ type: 'progress', taskId, status: 'completed', progress: 100 });
    post({ type: 'result', taskId, blob: resultBlob, params });
    log(taskId, 'Завершено');

  } catch (err: any) {
    log(taskId, `ОШИБКА: ${err.message}`);
    console.error(err);
    
    if (err.message === 'TASK_CANCELLED') {
      post({ type: 'progress', taskId, status: 'cancelled', progress: 0 });
    } else {
      post({ type: 'error', taskId, message: err.message });
    }
  }
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;

  if (msg.type === 'start') {
    currentTaskId = msg.taskId;
    isCancelled = false;
    log(msg.taskId, 'Получена задача');
    await processTask(msg.taskId, msg.image);
  } 
  
  if (msg.type === 'cancel') {
    if (currentTaskId === msg.taskId) {
      log(msg.taskId, 'Отмена');
      isCancelled = true;
    }
  }
};