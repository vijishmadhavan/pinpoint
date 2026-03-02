# Image Analysis

## Tools
- **read_file(path)** → You SEE the image visually. Best for single images — describe, answer questions, read text directly.
- **ocr(path, folder?)** → Tesseract OCR (125 languages, auto-detects script). Extracts text from images or scanned PDFs. Batch: pass folder for all at once.
- **detect_objects(path, object)** → Moondream. Find objects with bounding boxes (x_min, y_min, x_max, y_max). Use for cropping non-human objects.
- **search_images_visual(folder, query)** → SigLIP2. Finds images matching text description across a folder. Cached after first run.

## When to use what
- Single image → read_file (you see it directly)
- Find images by description → search_images_visual (SigLIP2)
- Extract text from image → ocr
- Crop an object from image → detect_objects (bounding box) + crop_image
- Face matching/cropping → face analysis tools (InsightFace)
