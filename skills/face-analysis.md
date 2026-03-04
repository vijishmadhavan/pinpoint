# Face Analysis (InsightFace)

## What InsightFace can do
Face geometry: count, locate, match, compare faces. Age, gender, head pose.
Persistent face memory: remember and auto-recognize known people.

## What InsightFace cannot do
Cannot describe scenes, clothing, activities, objects, or context. For that use read_file (Gemini sees images inline).

## Tools
- **detect_faces(image_path)** → Full face analysis. Returns per face: bounding box, confidence, estimated age, gender (M/F), head pose (pitch/yaw/roll). Auto-recognizes known faces (adds `name` field if matched). Cached — repeat calls instant.
- **crop_face(image_path, face_idx)** → Crop face with 25% padding, save to temp file. Returns: cropped image path + face metadata. Send via send_file to show user.
- **find_person(reference_image, folder)** → Find matching faces in folder. Reference must have exactly 1 face. Returns: matching image paths sorted by similarity, with age/gender of each match. First scan slow (~5-10s/100 images), cached after.
- **find_person_by_face(reference_image, face_idx, folder)** → Same as find_person but for multi-face reference. Specify which face by face_idx.
- **count_faces(image_path)** → Quick face count with demographics: total faces, male/female counts, age range (min/max/avg).
- **compare_faces(image_path_1, face_idx_1?, image_path_2, face_idx_2?)** → Compare two faces. Returns: similarity score (0-1), same_person boolean, confidence (high/medium/low). face_idx defaults to 0.
- **remember_face(image_path, face_idx?, name)** → Save a face for persistent recognition. After this, detect_faces auto-identifies this person. Multiple photos of the same person improve accuracy (different angles/lighting). face_idx defaults to 0.
- **forget_face(name)** → Delete all saved face data for a person (case-insensitive). They will no longer be auto-recognized.

## Face Search Flow
1. User sends photo → detect_faces
2. One face → ask which folder → find_person
3. Multiple faces → crop_face each → send_file crops → user picks → find_person_by_face
4. Zero faces → ask for clearer photo

## Face Memory Flow
1. User sends photo + "remember this as X" → detect_faces → remember_face(image_path, face_idx, "X")
2. Later: user sends any photo → detect_faces auto-returns `name: "X"` if recognized
3. "Forget X" → forget_face("X") → deleted
4. Multiple photos of same person → remember_face each → better recognition accuracy
