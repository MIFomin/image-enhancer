import { defineConfig } from 'vite';

export default defineConfig({
  base: '/image-enhancer/',
  
  build: {
    target: 'es2020',
    outDir: 'dist',
  },
  
  worker: {
    format: 'es',
  },
});