# Google Workspace

## Can
- Send emails via Gmail (with optional file attachment)
- Search Gmail inbox (same syntax as Gmail search bar)
- Check unread inbox summary (triage)
- List upcoming calendar events (today, this week, N days)
- Create calendar events with attendees and location
- List and search Google Drive files
- Upload local files to Google Drive

## Cannot
- Read full email bodies (only snippets/metadata)
- Delete or modify existing emails
- Modify or delete calendar events
- Download files from Drive to local
- Manage Drive permissions or sharing
- Access Google Sheets/Docs content directly

## Patterns
- Email with attachment: user says "email this PDF to john" → first find the file (search/list), then gmail_send with attach param
- Calendar: times must be ISO 8601 with timezone offset (IST = +05:30). User says "3pm tomorrow" → convert to 2026-03-07T15:00:00+05:30
- Drive search: empty query returns recent files sorted by modified time
- Gmail search: supports from:, to:, subject:, has:attachment, after:YYYY/MM/DD, before:, is:unread

## Don't
- Send email without confirming recipient and content with user first
- Create calendar events without confirming time with user
- Upload files to Drive without user explicitly asking
