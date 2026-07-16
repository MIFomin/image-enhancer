import { copyFileSync, mkdirSync, existsSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const rootDir = join(__dirname, '..');
const sourceDir = join(rootDir, 'node_modules', 'onnxruntime-web', 'dist');
const targetDir = join(rootDir, 'public');

console.log('Копирование WASM файлов для ONNX Runtime...');

if (!existsSync(sourceDir)) {
  console.error('Папка node_modules/onnxruntime-web/dist не найдена!');
  console.error('Выполните: npm install');
  process.exit(1);
}

if (!existsSync(targetDir)) {
  mkdirSync(targetDir, { recursive: true });
}

const wasmFiles = readdirSync(sourceDir).filter(file => file.endsWith('.wasm'));

if (wasmFiles.length === 0) {
  console.warn('WASM файлы не найдены в node_modules/onnxruntime-web/dist/');
  process.exit(0);
}

let copiedCount = 0;
for (const file of wasmFiles) {
  const sourcePath = join(sourceDir, file);
  const targetPath = join(targetDir, file);
  
  try {
    copyFileSync(sourcePath, targetPath);
    console.log(`  ${file}`);
    copiedCount++;
  } catch (err) {
    console.error(`Ошибка копирования ${file}:`, err.message);
  }
}

console.log(`\nСкопировано ${copiedCount} WASM файлов в public/`);