# PDF Tools

## Can do
Merge, split, render pages to images, combine images into PDF, extract tables. Fast structural operations.

## Cannot do
Cannot read/extract text from PDFs. For that use read_document (indexed) or read_file (direct).

## Tools
- **merge_pdf(paths, output_path)** → Combine multiple PDFs into one. Preserves all pages. Use for: combining invoices, reports, certificates.
- **split_pdf(path, pages, output_path)** → Extract specific pages from a PDF. Pages format: "1-5", "3,7,10", "1-3,5,8-10". Use for: extracting specific pages, removing pages.
- **pdf_to_images(path, pages?, dpi?, output_folder?)** → Render PDF pages as PNG images. Default 150 DPI.
- **images_to_pdf(paths, output_path)** → Combine multiple images into a single PDF.
- **extract_tables(path, pages?)** → Extract structured tables from PDF. Returns headers + rows. Works on native PDFs (invoices, reports, spreadsheets). Not for scanned PDFs — use OCR first.

## Notes
- Uses PyMuPDF — fast, no quality loss
- Page numbers are 1-based (first page = 1)
- Confirm with user before overwriting existing files
