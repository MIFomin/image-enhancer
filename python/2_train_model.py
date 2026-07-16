"""
Обучение модели для улучшения изображений
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("Обучение модели Image Enhancer")
print("=" * 60)

# Загружаем датасет
print("\nЗагрузка датасета...")
if not os.path.exists('data/flowers_dataset.npy'):
    print("ОШИБКА: Сначала запустите 1_download_dataset.py")
    exit(1)

images = np.load('data/flowers_dataset.npy')
print(f"Загружено {images.shape[0]} изображений")

# Функция для применения искажений
def apply_distortions(image):
    brightness_distort = np.random.uniform(-0.3, 0.3)
    contrast_distort = np.random.uniform(0.6, 1.4)
    saturation_distort = np.random.uniform(0.4, 1.6)
    
    distorted = image.copy()
    distorted = distorted + brightness_distort
    distorted = (distorted - 0.5) * contrast_distort + 0.5
    
    gray = np.dot(distorted, [0.299, 0.587, 0.114])[..., None]
    distorted = gray + (distorted - gray) * saturation_distort
    distorted = np.clip(distorted, 0, 1)
    
    target_brightness = -brightness_distort * 0.8
    target_contrast = 1.0 / contrast_distort
    target_saturation = 1.0 / saturation_distort
    
    return distorted, [target_brightness, target_contrast, target_saturation]

# Генерируем датасет пар
print("\nГенерация пар (искаженное, параметры коррекции)...")
X_train = []
y_train = []

for i, img in enumerate(images):
    distorted, params = apply_distortions(img)
    X_train.append(distorted)
    y_train.append(params)
    
    if (i + 1) % 1000 == 0:
        print(f"Обработано {i + 1} изображений")

X_train = np.array(X_train)
y_train = np.array(y_train)

split = int(0.8 * len(X_train))
X_tr, X_val = X_train[:split], X_train[split:]
y_tr, y_val = y_train[:split], y_train[split:]

print(f"\nTrain: {X_tr.shape[0]} изображений")
print(f"Validation: {X_val.shape[0]} изображений")

# Создаём модель
print("\nСоздание модели...")
def create_model(input_shape=(64, 64, 3)):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(3, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

model = create_model()
model.summary()

# Обучаем
print("\n" + "=" * 60)
print("Начинаем обучение...")
print("=" * 60)

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=1
)

# Графики обучения
print("\nПостроение графиков...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Mean Absolute Error')

plt.tight_layout()
os.makedirs('data', exist_ok=True)
plt.savefig('data/training_history.png', dpi=150)
print("Графики сохранены в data/training_history.png")
plt.show()

# Тестирование на примерах
print("\n" + "=" * 60)
print("Тестирование на случайных примерах")
print("=" * 60)

num_tests = 4
fig, axes = plt.subplots(num_tests, 3, figsize=(15, 4 * num_tests))

for i in range(num_tests):
    idx = np.random.randint(0, len(X_val))
    test_img = X_val[idx:idx+1]
    true_params = y_val[idx]
    
    predicted = model.predict(test_img, verbose=0)[0]
    
    corrected = test_img[0].copy()
    corrected = corrected + predicted[0]
    corrected = (corrected - 0.5) * predicted[1] + 0.5
    gray = np.dot(corrected, [0.299, 0.587, 0.114])[..., None]
    corrected = gray + (corrected - gray) * predicted[2]
    corrected = np.clip(corrected, 0, 1)
    
    axes[i, 0].imshow(test_img[0])
    axes[i, 0].set_title("Искаженное")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(corrected)
    axes[i, 1].set_title(f"Коррекция (предсказание)\nB={predicted[0]:.2f}, C={predicted[1]:.2f}, S={predicted[2]:.2f}")
    axes[i, 1].axis('off')
    
    gt_corrected = test_img[0].copy()
    gt_corrected = gt_corrected + true_params[0]
    gt_corrected = (gt_corrected - 0.5) * true_params[1] + 0.5
    gray = np.dot(gt_corrected, [0.299, 0.587, 0.114])[..., None]
    gt_corrected = gray + (gt_corrected - gray) * true_params[2]
    gt_corrected = np.clip(gt_corrected, 0, 1)
    
    axes[i, 2].imshow(gt_corrected)
    axes[i, 2].set_title(f"Ground Truth\nB={true_params[0]:.2f}, C={true_params[1]:.2f}, S={true_params[2]:.2f}")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig('data/test_results.png', dpi=150)
print("Результаты тестирования сохранены в data/test_results.png")
plt.show()

# Сохраняем модель в НОВОМ формате .keras
print("\nСохранение модели...")
os.makedirs('models', exist_ok=True)
model.save('models/enhancer_model.keras')
print("Модель сохранена в models/enhancer_model.keras")

print("\n" + "=" * 60)
print("Готово! Теперь запустите 3_export_model.py")
print("=" * 60)