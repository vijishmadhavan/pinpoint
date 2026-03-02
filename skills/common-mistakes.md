# Common Mistakes to Avoid

- **Don't** process 10+ images one by one with read_file. **Instead** use caption_image or query_image with folder param (Moondream, cheap batch processing).
- **Don't** retry a failed search with the same query. **Instead** reformulate with synonyms, broader terms, or try search_facts.
- **Don't** process a whole folder without checking its size first. **Instead** call list_files to survey the folder, then decide.
- **Don't** call search_documents AND search_facts with the same query. **Instead** pick one — search_documents for full text, search_facts for quick factual lookups.
- **Don't** skip analyze_data(operation=columns) and go straight to filter/groupby. **Instead** always call columns FIRST to see sheet names, column types, and sample values.
- **Don't** call detect_faces or caption_image one image at a time in a loop. **Instead** pass the folder parameter to process all images in one call.
- **Don't** call a file tool without checking the path exists. **Instead** if unsure about the exact path, use list_files on the parent folder first.
- **Don't** keep calling tools when you already have enough info to answer. **Instead** stop and give the user your answer — more rounds waste time.
