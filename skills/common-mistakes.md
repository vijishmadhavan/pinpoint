# Common Mistakes to Avoid

- **Don't** jump straight to search_images_visual for finding photos. **Instead** try search_documents(query, file_type="image", folder=...) FIRST — indexed captions are free and instant. Only use search_images_visual if search_documents returns no results.
- **Don't** process 10+ images one by one with read_file. **Instead** use search_images_visual with folder param (SigLIP2, cached batch processing).
- **Don't** retry a failed search with the same query. **Instead** reformulate with synonyms, broader terms, or try search_facts.
- **Don't** process a whole folder without checking its size first. **Instead** call list_files to survey the folder, then decide.
- **Don't** call search_documents AND search_facts with the same query. **Instead** pick one — search_documents for full text, search_facts for quick factual lookups.
- **Don't** skip analyze_data(operation=columns) and go straight to filter/groupby. **Instead** always call columns FIRST to see sheet names, column types, and sample values.
- **Don't** call detect_faces one image at a time in a loop. **Instead** pass the folder parameter to process all images in one call.
- **Don't** call a file tool without checking the path exists. **Instead** if unsure about the exact path, use list_files on the parent folder first.
- **Don't** keep calling tools when you already have enough info to answer. **Instead** stop and give the user your answer — more rounds waste time.
- **Don't** manually inspect images (read_file/resize_image/ocr) after getting search_images_visual results. **Instead** trust the visual search results — they are AI-analyzed and reliable enough to categorize, group, or answer directly.
- **Don't** score photos one by one with score_photo in a loop. **Instead** use cull_photos to batch-score and auto-separate keepers from rejects.
- **Don't** classify photos one by one with search_images_visual (top N). **Instead** use group_photos to classify ALL photos via Gemini vision into category subfolders.
- **Don't** filter only .xlsx when user asks for "excel files" or "spreadsheets". **Instead** include .csv, .xls, .xlsx, and .tsv — users mean all tabular data files.
