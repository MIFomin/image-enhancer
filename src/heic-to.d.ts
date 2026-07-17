declare module 'heic-to' {
  interface HeicToOptions {
    blob: Blob;
    type?: 'image/jpeg' | 'image/png' | 'bitmap';
    quality?: number;
    options?: {
      imageOrientation?: string;
    };
  }
  
  function heicTo(options: HeicToOptions): Promise<Blob>;
  function isHeic(file: Blob): Promise<boolean>;
  
  export { heicTo, isHeic };
}