# Visual Search

## Can do
Search images in a folder by text description. Uses AI vision embeddings (SigLIP2) for text-to-image similarity. First call embeds folder (~180ms/image GPU), cached for instant re-queries. Returns ranked results with similarity scores.

## Cannot do
Cannot identify specific people — use face search (find_person, detect_faces) for that. Cannot search inside documents or PDFs. Cannot search across multiple folders in one call.

## Tools
- **search_images_visual(folder, query, limit?)** → Find images matching a text description. Returns top matches ranked by visual similarity. Use for: "find sunset photos", "photos with cake", "outdoor group shots". First call on a folder takes time (embedding), repeat queries are instant.
- **search_images_visual(folder, queries=[], limit?)** → Batch: multiple queries in one call. E.g. queries=["dancing", "flowers", "group photo"]. Folder is embedded once, all queries run against the same cache.

## Notes
- Embedding is cached per folder — first search takes ~18s/100 images, subsequent searches are instant (15ms)
- Use queries array for batch search — more efficient than multiple single calls
- Works best with descriptive queries: "bride and groom", "flowers", "people dancing"
- For people identification, use face search instead
- After finding images, can send_file to share or read_file to see them
