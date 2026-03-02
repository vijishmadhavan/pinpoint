# Batch Awareness

## Preprocessing (automatic, originals never touched)
- Images >1024px: auto-resized to max 1024px before Moondream/InsightFace
- Large batch OCR: renders at 200 DPI instead of 300
- All resizing happens in memory — original files stay untouched

## Known Speeds (per item)

| Operation | GPU | CPU |
|---|---|---|
| Visual search embed (SigLIP2) | ~200ms/image | ~400ms/image |
| Face detection (InsightFace) | ~0.3s/image | ~1.3s/image |
| Image caption (Moondream) | ~0.85s/image | ~5s/image |
| OCR scanned PDF (Tesseract) | — | ~0.5s/page |
| Digital PDF text extract | — | ~0.01s/page |

## Batch Tools Available
- search_images_visual(folder, query) — embeds folder once, cached for instant re-queries
- detect_faces(folder=) / count_faces(folder=) / caption_image(folder=) / ocr(folder=)
- batch_move(sources[], destination) — move multiple files in one call
