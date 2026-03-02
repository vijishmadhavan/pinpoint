# Archive Tools

## Can do
Compress files/folders into .zip. Extract .zip archives.

## Cannot do
Cannot handle .rar, .7z, .tar.gz formats. Only .zip supported.

## Tools
- **compress_files(paths, output_path)** → Zip files or folders into a .zip archive. Can include multiple files and entire folders. Use for: bundling files for sharing, backup, organizing.
- **extract_archive(path, output_path?)** → Extract a zip archive. If no output_path, extracts to folder with same name as archive. Use for: unzipping received files.

## Notes
- Supports .zip format
- Folders are included recursively
- Confirm with user before extracting (may overwrite)
