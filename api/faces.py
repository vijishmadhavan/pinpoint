"""Face recognition router — detect, crop, find, compare, remember, forget, recognize."""

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.helpers import _check_safe, _get_conn, _get_images_in_folder

router = APIRouter()


# --- Pydantic models ---


class DetectFacesRequest(BaseModel):
    image_path: str = None
    folder: str = None


class CropFaceRequest(BaseModel):
    image_path: str
    face_idx: int


class FindPersonRequest(BaseModel):
    reference_image: str
    folder: str
    threshold: float = 0.4


class FindPersonByFaceRequest(BaseModel):
    reference_image: str
    face_idx: int
    folder: str
    threshold: float = 0.4


class CountFacesRequest(BaseModel):
    image_path: str = None
    folder: str = None
    paths: list = None


class CompareFacesRequest(BaseModel):
    image_path_1: str
    face_idx_1: int = 0
    image_path_2: str
    face_idx_2: int = 0


class RememberFaceRequest(BaseModel):
    image_path: str
    face_idx: int = 0
    name: str


class ForgetFaceRequest(BaseModel):
    name: str


class RecognizeFacesRequest(BaseModel):
    image_path: str


# --- Endpoints ---


@router.post("/detect-faces")
def detect_faces_endpoint(req: DetectFacesRequest):
    """Detect faces in an image or all images in a folder."""
    from face_search import detect_faces

    conn = _get_conn()

    if req.folder:
        images = _get_images_in_folder(req.folder)
        if not images:
            raise HTTPException(status_code=404, detail=f"No images found in: {req.folder}")
        _BATCH_CAP = 100  # prevent multi-minute blocking on huge folders
        capped = len(images) > _BATCH_CAP
        images = images[:_BATCH_CAP]
        results = {}
        for img_path in images:
            result = detect_faces(img_path, conn)
            if not (isinstance(result, dict) and "error" in result):
                results[os.path.basename(img_path)] = {"faces": result, "face_count": len(result)}
        resp = {"folder": os.path.abspath(req.folder), "images_processed": len(results), "results": results}
        if capped:
            resp["_hint"] = f"Capped at {_BATCH_CAP} images. Process remaining in separate calls."
        else:
            resp["_hint"] = f"Face detection complete for {len(results)} images. Report results directly."
        return resp

    result = detect_faces(req.image_path, conn)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    resp = {"image_path": os.path.abspath(req.image_path), "faces": result, "face_count": len(result)}
    if len(result) > 0:
        resp["_hint"] = (
            "Use find_person(ref_image, folder) to search for this person in other photos. Use crop_face to isolate a specific face."
        )
    return resp


@router.post("/crop-face")
def crop_face_endpoint(req: CropFaceRequest):
    """Crop a specific face from an image. Returns path to cropped image."""
    from face_search import crop_face

    conn = _get_conn()
    result = crop_face(req.image_path, req.face_idx, conn)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/find-person")
def find_person_endpoint(req: FindPersonRequest):
    """Find photos matching a reference face in a folder."""
    from face_search import find_person

    conn = _get_conn()
    result = find_person(req.reference_image, req.folder, conn, req.threshold)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    if isinstance(result, list):
        result = {
            "matches": result,
            "count": len(result),
            "_hint": f"{len(result)} photo(s) with matching face. Report or send these.",
        }
    return result


@router.post("/find-person-by-face")
def find_person_by_face_endpoint(req: FindPersonByFaceRequest):
    """Find photos matching a specific face (by index) from a reference image."""
    from face_search import find_person_by_face

    conn = _get_conn()
    result = find_person_by_face(req.reference_image, req.face_idx, req.folder, conn, req.threshold)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    if isinstance(result, list):
        result = {
            "matches": result,
            "count": len(result),
            "_hint": f"{len(result)} photo(s) with matching face. Report or send these.",
        }
    return result


@router.post("/count-faces")
def count_faces_endpoint(req: CountFacesRequest):
    """Count faces in an image, a list of images, or all images in a folder."""
    from face_search import count_faces

    conn = _get_conn()

    # Batch by paths list (e.g. from visual search results)
    if req.paths:
        results = {}
        for img_path in req.paths:
            abs_path = os.path.abspath(img_path)
            _check_safe(abs_path)
            if os.path.isfile(abs_path):
                result = count_faces(abs_path, conn)
                if not (isinstance(result, dict) and "error" in result):
                    results[os.path.basename(abs_path)] = result
        return {"images_processed": len(results), "results": results}

    if req.folder:
        images = _get_images_in_folder(req.folder)
        if not images:
            raise HTTPException(status_code=404, detail=f"No images found in: {req.folder}")
        results = {}
        for img_path in images:
            result = count_faces(img_path, conn)
            if not (isinstance(result, dict) and "error" in result):
                results[os.path.basename(img_path)] = result
        resp = {"folder": os.path.abspath(req.folder), "images_processed": len(results), "results": results}
        resp["_hint"] = f"Face counts complete for {len(results)} images. Report results directly."
        return resp

    result = count_faces(req.image_path, conn)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    result["_hint"] = "Face count complete. Report the number directly."
    return result


@router.post("/compare-faces")
def compare_faces_endpoint(req: CompareFacesRequest):
    """Compare two faces to check if they're the same person."""
    from face_search import compare_faces

    conn = _get_conn()
    result = compare_faces(req.image_path_1, req.face_idx_1, req.image_path_2, req.face_idx_2, conn)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    result["_hint"] = "Score >0.6 = likely same person, 0.4-0.6 = uncertain, <0.4 = different person."
    return result


# --- Face memory (Segment 18V: persistent face recognition) ---


@router.post("/remember-face")
def remember_face_endpoint(req: RememberFaceRequest):
    """Save a face for future recognition."""
    from face_search import remember_face

    conn = _get_conn()
    result = remember_face(req.image_path, req.face_idx, req.name, conn)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    result["_hint"] = (
        f"Face saved as '{req.name}'. detect_faces will now auto-recognize this person. Add more photos of them for better accuracy."
    )
    return result


@router.post("/forget-face")
def forget_face_endpoint(req: ForgetFaceRequest):
    """Delete all saved face data for a person."""
    from face_search import forget_face

    conn = _get_conn()
    result = forget_face(req.name, conn)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/recognize-faces")
def recognize_faces_endpoint(req: RecognizeFacesRequest):
    """Recognize known faces in an image."""
    from face_search import recognize_faces

    conn = _get_conn()
    result = recognize_faces(req.image_path, conn)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
