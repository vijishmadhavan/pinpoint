# Image Analysis

## Tools
- **read_file(path)** → You SEE the image visually. Best for single images — describe, answer questions, read text directly.
- **ocr(path, folder?)** → Tesseract OCR (125 languages, auto-detects script). Extracts text from images or scanned PDFs. Batch: pass folder for all at once.
- **search_images_visual(folder, query)** → SigLIP2. Finds images matching text description across a folder. Cached after first run.
- **image_metadata(path, folder?)** → EXIF data: date taken, camera, GPS, lens, aperture, ISO, dimensions. Batch: pass folder for all images. Use for timelines, camera info, location questions.

## When to use what
- Single image → read_file (you see it directly)
- Find images by description → search_images_visual (SigLIP2)
- Extract text from image → ocr
- Face matching/cropping → face analysis tools (InsightFace)
