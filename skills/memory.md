# Persistent Memory

## Can do
Save and recall any facts across sessions. Persists forever (survives restarts, idle resets).
Smart dedup: won't save duplicates, merges related facts, handles contradictions automatically.

## Cannot do
Cannot store files or images — only text facts.

## Tools
- **memory_save(fact, category?)** → Save a fact. Categories: people, places, preferences, professional, health, plans, general. Automatically detects duplicates (skips), merges related facts, supersedes contradictions.
- **memory_search(query)** → Search saved memories by keyword.
- **memory_delete(id)** → Delete a memory by ID (when you have the ID).
- **memory_forget(description)** → Forget a memory by description — no ID needed. Searches and deletes best match.

## What to remember
- Personal preferences (food, products, activities, entertainment)
- Important personal details (names, relationships, birthdays, addresses)
- Plans and intentions (upcoming events, goals, travel)
- Professional details (job, company, skills, career goals)
- Health and wellness (allergies, diet, exercise habits)
- Activity and service preferences (restaurants, apps, brands)

## What NOT to remember
- Greetings ("Hi", "Hello") — no useful fact
- Temporary context ("search for X", "find my file") — not a lasting fact
- Things already in their documents — use search_documents instead

## Toggle
- On by default. User disables with `/memory off`, re-enables with `/memory on`.
- `/memory` shows status and saved memories.
- When off, memory tools return "Memory is disabled."
