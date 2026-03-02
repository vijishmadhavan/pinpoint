# Video Search

## Can do
Search inside a video by text description. Extracts frames using ffmpeg, embeds with SigLIP2 vision AI, finds moments matching the query. Returns timestamps with match scores. Cached after first run — repeat queries on same video are instant. Can extract specific frames as images for sending.

## Cannot do
Cannot play or stream video. Cannot search audio/speech content. Cannot search across multiple videos in one call.

## Tools
- **search_video(video_path, query, fps?, limit?)** → Find moments in a video matching a text description. Returns timestamps ranked by visual similarity.
- **extract_frame(video_path, seconds)** → Extract a single frame at a timestamp as an image. Use after search_video to get the actual frame for sending.

## Notes
- First search on a video extracts and embeds all frames (takes time), subsequent searches are instant
- Default 1 frame/second. Use lower fps (0.5) for long videos, higher (2) for short clips
- After finding matching timestamps, use extract_frame + send_file to share the frame
- Supports: mp4, mkv, avi, mov, wmv, flv, webm
