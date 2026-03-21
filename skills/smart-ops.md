# Smart Operations

## Can do
Find duplicate files by content hash. Bulk rename files with regex patterns.

## Cannot do
Cannot find visually similar images or duplicate photos. For visual search use search_images_visual (SigLIP2). For face matching use compare_faces.

## Tools
- **find_duplicates(folder)** → Scan folder recursively to find duplicate files by content. Returns groups of identical files. Use for: cleanup, freeing disk space, finding copies.
- **batch_rename(folder, pattern, replace)** → Rename files matching a regex pattern. pattern is a regex, replace is the replacement string. Use for: bulk renaming, fixing naming conventions, adding prefixes/suffixes.

## Notes
- find_duplicates uses content hash — catches files with different names but same content
- batch_rename only renames files (not folders), won't overwrite existing files
- batch_rename shows a preview of changes before applying
- Regex examples: "IMG_" → "" (remove prefix), "(.+)\\.jpeg" → "\\1.jpg" (fix extension)

## Post-Operation Follow-Ups
- After `find_duplicates`, follow-up requests like "show me the larger one", "send the original", or "open that duplicate" should stay within the returned duplicate groups first.
- After `batch_rename`, follow-ups like "open the renamed file", "send the cleaned one", or "show what changed" should use the rename result set or destination folder first instead of starting a broad search.
