"""
Скачивание датасета oxford_flowers102
"""
import numpy as np
from PIL import Image
import os
import urllib.request
import tarfile
import scipy.io as sio
import matplotlib.pyplot as plt

print("=" * 60)
print("Скачивание датасета Oxford Flowers 102")
print("=" * 60)

# Создаём папку data
os.makedirs('data', exist_ok=True)
os.makedirs('data/flowers', exist_ok=True)

# URL для скачивания
images_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
splits_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

# Функция для скачивания файла
def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"Файл уже существует: {dest_path}")
        return
    
    print(f"Скачивание: {url}")
    print("Это может занять несколько минут...")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rПрогресс: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
    
    urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
    print("\nСкачивание завершено!")

# Скачиваем файлы
download_file(images_url, 'data/102flowers.tgz')
download_file(labels_url, 'data/imagelabels.mat')
download_file(splits_url, 'data/setid.mat')

# Распаковываем изображения
print("\nРаспаковка изображений...")
images_dir = 'data/flowers/jpg'
if not os.path.exists(images_dir):
    with tarfile.open('data/102flowers.tgz', 'r:gz') as tar:
        tar.extractall('data/flowers')
    print("Распаковка завершена!")
else:
    print("Изображения уже распакованы")

# Загружаем метки
print("\nЗагрузка меток...")
labels = sio.loadmat('data/imagelabels.mat')['labels'][0]
setid = sio.loadmat('data/setid.mat')
train_ids = setid['trnid'][0] - 1  # 0-indexed
val_ids = setid['valid'][0] - 1
test_ids = setid['tstid'][0] - 1

print(f"Train: {len(train_ids)}, Validation: {len(val_ids)}, Test: {len(test_ids)}")

# Функция для загрузки и предобработки изображения
def load_and_preprocess(image_id, target_size=64):
    img_path = os.path.join(images_dir, f'image_{image_id + 1:05d}.jpg')
    img = Image.open(img_path).convert('RGB')
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array

# Загружаем все изображения
print("\nЗагрузка всех изображений в память (это может занять минуту)...")
all_ids = np.concatenate([train_ids, val_ids, test_ids])
images = []

for i, img_id in enumerate(all_ids):
    img = load_and_preprocess(img_id)
    images.append(img)
    
    if (i + 1) % 1000 == 0:
        print(f"Загружено {i + 1} изображений")

images = np.array(images)
print(f"\nИтого загружено: {images.shape[0]} изображений")
print(f"Размер: {images.shape}")

# Сохраняем датасет
np.save('data/flowers_dataset.npy', images)
print(f"\nДатасет сохранён в data/flowers_dataset.npy")

# Визуализация
print("\nВизуализация примеров из датасета:")
fig, axes = plt.subplots(3, 6, figsize=(15, 7))
for i in range(18):
    idx = np.random.randint(0, len(images))
    axes[i // 6, i % 6].imshow(images[idx])
    axes[i // 6, i % 6].axis('off')
plt.tight_layout()
plt.savefig('data/dataset_preview.png', dpi=100)
print("Превью сохранено в data/dataset_preview.png")
plt.show()

print("\n" + "=" * 60)
print("Готово! Теперь запустите 2_train_model.py")
print("=" * 60)