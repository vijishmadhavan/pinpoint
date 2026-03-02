# Image Tools

## Can do
Resize, convert formats (HEIC/JPG/PNG/WebP), crop, compress images.

## Cannot do
Cannot understand image content. For that use caption_image (describe scene) or detect_faces (find faces).

## Tools
- **resize_image(path, width?, height?, quality?, output_path?)** → Resize or compress. Set width OR height to maintain aspect ratio, or both for exact size. quality 1-100 (default 85). If no output_path, overwrites original.
- **convert_image(path, format, output_path?, quality?)** → Convert format. Supports: jpg, png, webp, bmp. Handles HEIC (iPhone photos). Output defaults to same name with new extension.
- **crop_image(path, x, y, width, height, output_path?)** → Crop to rectangle. x,y = top-left corner, width,height = crop size in pixels.

## Notes
- For HEIC→JPG conversion (iPhone photos), convert_image handles it automatically
- resize_image with just quality parameter = compress without resizing
- Use file_info or read_file first to check image dimensions if needed
