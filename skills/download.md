# Download

## Can do
Download any file from HTTP/HTTPS URLs.

## Cannot do
Cannot browse websites or extract web page content. Only direct file downloads.

## Tools
- **download_url(url, save_path?)** → Download file from any URL. If no save_path, saves to Downloads/Pinpoint/ with original filename. Use for: downloading files shared as links, fetching web resources.

## Notes
- Only http:// and https:// URLs
- 60 second timeout
- Saves to Downloads/Pinpoint/ by default
- After download, can send via send_file or index via index_file
