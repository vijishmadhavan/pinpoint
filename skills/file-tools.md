# File Tools

## Can do
Read, list, move, copy, delete, send files. read_file lets you SEE images visually.

## Cannot do
Cannot analyze data in spreadsheets (use analyze_data), cannot search across files (use search_documents).

## Tools
- **read_file(path)** → Read any file from disk. Images: you SEE them visually (base64). Documents (PDF/DOCX/TXT/XLSX): extracted text. Use for bills, receipts, business cards, photos where you need to see actual content.
- **list_files(folder, sort_by?, filter_ext?, filter_type?)** → List folder contents. Sort: name/date/size. Filter by extension (.pdf) or category (image/document/spreadsheet/video/audio/archive).
- **file_info(path)** → File metadata: size, dates, extension, indexed status, document ID.
- **move_file(source, destination, copy?)** → Move or rename a single file.
- **batch_move(sources[], destination, is_copy?)** → Move or copy multiple files to a folder in one call. Much faster than calling move_file repeatedly.
- **copy_file(source, destination)** → Copy a file or folder to a new location.
- **create_folder(path)** → Create directory (parents too).
- **delete_file(path)** → Delete a file. ALWAYS ask user confirmation. Cannot delete folders.
- **send_file(path, caption?)** → Send file to user on WhatsApp. Images sent as photos, rest as documents. ONLY when user asks.
- **index_file(path)** → Index a specific file into the search database on demand. Extracts text, stores in FTS5. Use when user sends a new file and wants it searchable.

## Notes
- File paths must be absolute (not relative)
- File paths come from search_documents, list_files, or search_images_visual results
