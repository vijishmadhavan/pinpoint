# Search

## Can do
Find indexed documents by text content. Full-text search with stemming across all file types.

## Cannot do
Cannot search by visual content. For finding images by description use search_images_visual (SigLIP2). For seeing a specific image use read_file.

## Tools
- **search_documents(query, file_type?, folder?)** → Search all indexed files. Returns: document IDs, paths, titles, relevance scores, text snippets. Filters: file_type (pdf/docx/xlsx/image/txt/csv/epub), folder path prefix.
- **search_facts(query)** → Search extracted facts from documents. Quick factual lookups (names, dates, amounts). Use for specific questions before full-text search.
- **read_document_overview(document_id, query?)** → Compact overview of a document: metadata, top sections, extracted facts. Use BEFORE read_document to save tokens. Pass the user's query to rank relevant sections first.
- **read_document(document_id)** → Read full indexed text of a document by ID. Use only after overview when you need the complete text (summarization, comparison, translation).
- **search_history(query)** → Search past conversation messages from previous sessions. Returns: matching messages with timestamps.

## Auto-Indexing
- **watch_folder(folder)** → Start watching a folder. New/modified files auto-indexed. Persists across restarts.
- **unwatch_folder(folder)** → Stop watching a folder.
- **list_watched()** → Show all watched folders.

## Notes
- search_documents uses FTS5 with porter stemming — "invoices" matches "invoice"
- After search, use read_document_overview first — it shows the most relevant sections without loading the full text
- Use read_document only when you need the complete text (summarization, full comparison, translation)
- For finding images by description, use search_images_visual (much faster than captioning all)
- search_history is for things from older sessions not visible in current conversation
- Watched folders auto-index with 5s debounce (waits for file to finish writing)
