# Web Search

## Can do
Search the web for real-time information. Returns search results with titles, snippets, URLs. Can also read a specific URL for full content. Results are current and reliable — answer directly from them.

## Cannot do
Cannot access login-protected pages. Cannot execute JavaScript. Cannot submit forms.

## Tools
- **web_search(query)** → Search the web (LangSearch API). Returns results with titles, snippets, and URLs.
- **web_search(query, url)** → Read a specific URL for full content (e.g. to get details from a search result).
- **web_search(query, freshness)** → Filter by time: noLimit, day, week, month.
