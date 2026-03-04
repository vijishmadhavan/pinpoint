# Video Search

## Can do
Search inside a video by text description. Gemini analyzes the full video natively (up to 3 hours). Returns timestamps with match scores. Can extract specific frames as images for sending.

## Cannot do
Cannot play or stream video. Cannot search audio/speech content (use transcribe_audio/search_audio for that). Cannot search across multiple videos in one call.

## Tools
- **search_video(video_path, query, limit?)** → Find moments in a video matching a text description. Returns timestamps ranked by relevance.
- **extract_frame(video_path, seconds)** → Extract a single frame at a timestamp as an image. Use after search_video to get the actual frame for sending.

## Notes
- Gemini analyzes the full video in one call — no frame extraction needed
- After finding matching timestamps, use extract_frame + send_file to share the frame
- Supports: mp4, mkv, avi, mov, wmv, flv, webm
