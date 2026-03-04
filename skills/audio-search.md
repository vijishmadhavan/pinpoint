# Audio Search

## Can do
Transcribe audio files to text with timestamps. Search within audio for specific content (speech, sounds, topics). Gemini analyzes audio natively (up to 9.5 hours). Works with WhatsApp voice messages (.ogg).

## Cannot do
Cannot play or stream audio. Cannot edit or modify audio files. Cannot search across multiple audio files in one call.

## Tools
- **transcribe_audio(path)** → Full transcription with [MM:SS] timestamps. Use for voice memos, recordings, podcasts.
- **search_audio(audio_path, query, limit?)** → Find moments matching a query. Returns timestamps with relevance scores.

## Notes
- Supports: mp3, wav, flac, aac, ogg, wma, m4a, aiff
- WhatsApp voice messages are .ogg files — fully supported
- For long recordings, search_audio is more efficient than full transcription
