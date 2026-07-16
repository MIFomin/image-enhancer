"""
Экспорт модели в ONNX через CLI (универсальный способ)
Использует абсолютные пути и ищет модель в нескольких местах
"""
import tensorflow as tf
import onnx
import os
import shutil
import subprocess
import sys

# Получаем абсолютный путь к папке, где находится этот скрипт
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # image-enhancer/

print("=" * 60)
print("Экспорт модели в ONNX (через CLI)")
print("=" * 60)
print(f"Папка скрипта: {SCRIPT_DIR}")
print(f"Корень проекта: {PROJECT_ROOT}")

# Ищем модель в нескольких местах
model_path = None
search_paths = [
    # В папке python/models/ (где скрипт)
    os.path.join(SCRIPT_DIR, 'models', 'enhancer_model.keras'),
    os.path.join(SCRIPT_DIR, 'models', 'enhancer_model.h5'),
    # В папке models/ в корне проекта (если запускали 2_train из корня)
    os.path.join(PROJECT_ROOT, 'models', 'enhancer_model.keras'),
    os.path.join(PROJECT_ROOT, 'models', 'enhancer_model.h5'),
]

print("\nПоиск модели...")
for path in search_paths:
    if os.path.exists(path):
        model_path = path
        print(f"Найдена: {path}")
        break
    else:
        print(f"Не найдена: {path}")

if not model_path:
    print("\nОШИБКА: Модель не найдена!")
    print("Убедитесь, что вы запустили 2_train_model.py")
    print("И проверьте, где сохранилась модель:")
    print(f"  - {os.path.join(SCRIPT_DIR, 'models/')}")
    print(f"  - {os.path.join(PROJECT_ROOT, 'models/')}")
    exit(1)

print(f"\nЗагрузка модели из {model_path}...")
model = tf.keras.models.load_model(model_path, compile=False)
print("Модель загружена")

# Этап 1: Сохраняем в SavedModel формат
print("\n[Этап 1] Сохранение в SavedModel формат...")
saved_model_dir = os.path.join(SCRIPT_DIR, 'saved_model_temp')
if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)

@tf.function(input_signature=[tf.TensorSpec(shape=[1, 64, 64, 3], dtype=tf.float32)])
def serving_fn(x):
    return model(x, training=False)

tf.saved_model.save(
    model,
    saved_model_dir,
    signatures={'serving_default': serving_fn}
)
print(f"SavedModel сохранён в {saved_model_dir}")

# Этап 2: Конвертируем через CLI
print("\n[Этап 2] Конвертация SavedModel → ONNX через CLI...")
onnx_model_path = os.path.join(SCRIPT_DIR, 'models', 'enhancer_model.onnx')
saved_model_abs = os.path.abspath(saved_model_dir)

# Создаём папку models, если её нет
os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

# Получаем путь к текущему Python интерпретатору
python_exe = sys.executable
print(f"Используется Python: {python_exe}")

# Запускаем CLI конвертер
cmd = [
    python_exe, '-m', 'tf2onnx.convert',
    '--saved-model', saved_model_abs,
    '--output', onnx_model_path,
    '--opset', '13'
]

print(f"\nКоманда: {' '.join(cmd)}")
print("Выполнение (может занять 1-2 минуты)...\n")

result = subprocess.run(cmd, capture_output=True, text=True)

print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if result.returncode != 0:
    print(f"\nОШИБКА: CLI вернул код {result.returncode}")
    exit(1)

print("Конвертация завершена!")

# Удаляем временную папку
shutil.rmtree(saved_model_dir)
print("Временная папка SavedModel удалена")

# Проверяем результат
if not os.path.exists(onnx_model_path):
    print("ОШИБКА: ONNX файл не создан!")
    exit(1)

size = os.path.getsize(onnx_model_path)
print(f"\nРазмер модели: {size/1024:.1f} KB ({size/1024/1024:.2f} MB)")

# Проверяем валидность
print("\nПроверка валидности ONNX модели...")
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX модель валидна!")

# Информация о входах/выходах
print("\nИнформация о модели:")
for inp in onnx_model.graph.input:
    print(f"  Вход: {inp.name}, shape: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
for out in onnx_model.graph.output:
    print(f"  Выход: {out.name}, shape: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")

# Копируем в public/model (АБСОЛЮТНЫЙ ПУТЬ!)
print("\nКопирование в public/model/...")
public_model_dir = os.path.join(PROJECT_ROOT, 'public', 'model')
print(f"Целевая папка: {public_model_dir}")

if os.path.exists(public_model_dir):
    shutil.rmtree(public_model_dir)
os.makedirs(public_model_dir, exist_ok=True)

shutil.copy(onnx_model_path, os.path.join(public_model_dir, 'model.onnx'))
print(f"Модель скопирована в {public_model_dir}/model.onnx")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)
print(f"\nФайл модели: {public_model_dir}/model.onnx")
print("\nДалее выполните:")
print("  1. npm install onnxruntime-web")
print("  2. Обновите src/worker.ts")
print("  3. npm run dev")