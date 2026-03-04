"""File transformation & creation endpoints — write, generate, convert, compress, download, run."""

import os
import shutil

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.helpers import _check_safe

router = APIRouter()


# --- Write & Create (Segment 15B) ---


class WriteFileRequest(BaseModel):
    path: str
    content: str
    append: bool = False


@router.post("/write-file")
def write_file_endpoint(req: WriteFileRequest):
    """Create or write a text file."""
    path = os.path.abspath(req.path)
    _check_safe(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "a" if req.append else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(req.content)
    return {"success": True, "path": path, "size": os.path.getsize(path), "append": req.append}


class GenerateExcelRequest(BaseModel):
    path: str
    data: list  # list of dicts or list of lists
    columns: list | None = None
    sheet_name: str = "Sheet1"


@router.post("/generate-excel")
def generate_excel_endpoint(req: GenerateExcelRequest):
    """Create an Excel file from data."""
    import pandas as pd

    path = os.path.abspath(req.path)
    _check_safe(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df = pd.DataFrame(req.data, columns=req.columns)
        df.to_excel(path, sheet_name=req.sheet_name, index=False)
        return {"success": True, "path": path, "rows": len(df), "columns": list(df.columns)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create Excel: {e}")


class GenerateChartRequest(BaseModel):
    data: dict  # {"labels": [...], "values": [...]} or {"x": [...], "y": [...]}
    chart_type: str = "bar"  # bar, line, pie, scatter, hist
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    output_path: str | None = None


@router.post("/generate-chart")
def generate_chart_endpoint(req: GenerateChartRequest):
    """Generate a chart image using matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import tempfile

    import matplotlib.pyplot as plt

    output = (
        os.path.abspath(req.output_path)
        if req.output_path
        else os.path.join(tempfile.gettempdir(), "pinpoint_charts", f"chart_{int(__import__('time').time())}.png")
    )
    _check_safe(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = req.data.get("labels") or req.data.get("x", [])
        values = req.data.get("values") or req.data.get("y", [])

        if req.chart_type == "bar":
            ax.bar(labels, values)
        elif req.chart_type == "line":
            ax.plot(labels, values, marker="o")
        elif req.chart_type == "pie":
            ax.pie(values, labels=labels, autopct="%1.1f%%")
        elif req.chart_type == "scatter":
            ax.scatter(labels, values)
        elif req.chart_type == "hist":
            ax.hist(values, bins="auto")
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown chart type: {req.chart_type}. Use: bar, line, pie, scatter, hist"
            )

        if req.title:
            ax.set_title(req.title)
        if req.xlabel:
            ax.set_xlabel(req.xlabel)
        if req.ylabel:
            ax.set_ylabel(req.ylabel)
        plt.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        return {"success": True, "path": output}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chart generation failed: {e}")


# --- PDF Tools (Segment 15B) ---


class MergePdfRequest(BaseModel):
    paths: list[str]
    output_path: str


@router.post("/merge-pdf")
def merge_pdf_endpoint(req: MergePdfRequest):
    """Merge multiple PDFs into one."""
    import fitz  # PyMuPDF

    output = os.path.abspath(req.output_path)
    _check_safe(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    merged = fitz.open()
    for p in req.paths:
        p = os.path.abspath(p)
        _check_safe(p)
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail=f"File not found: {p}")
        doc = fitz.open(p)
        merged.insert_pdf(doc)
        doc.close()
    merged.save(output)
    total_pages = len(merged)
    merged.close()
    return {"success": True, "path": output, "total_pages": total_pages, "files_merged": len(req.paths)}


class SplitPdfRequest(BaseModel):
    path: str
    pages: str  # "1-5", "3,7,10", "1-3,5,8-10"
    output_path: str


@router.post("/split-pdf")
def split_pdf_endpoint(req: SplitPdfRequest):
    """Extract specific pages from a PDF."""
    import fitz

    path = os.path.abspath(req.path)
    _check_safe(path)
    output = os.path.abspath(req.output_path)
    _check_safe(output)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # Parse page specification
    page_nums = set()
    for part in req.pages.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            for i in range(int(start), int(end) + 1):
                page_nums.add(i)
        else:
            page_nums.add(int(part))

    doc = fitz.open(path)
    # Convert 1-based to 0-based
    zero_based = sorted(p - 1 for p in page_nums if 1 <= p <= len(doc))
    if not zero_based:
        doc.close()
        raise HTTPException(status_code=400, detail=f"No valid pages. PDF has {len(doc)} pages.")

    new_doc = fitz.open()
    for page_no in zero_based:
        new_doc.insert_pdf(doc, from_page=page_no, to_page=page_no)
    new_doc.save(output)
    extracted = len(new_doc)
    source_pages = len(doc)
    new_doc.close()
    doc.close()
    return {"success": True, "path": output, "pages_extracted": extracted, "source_pages": source_pages}


class PdfToImagesRequest(BaseModel):
    path: str
    pages: str | None = None  # "1,3,5" or "1-3" or None for all
    dpi: int = 150
    output_folder: str | None = None


@router.post("/pdf-to-images")
def pdf_to_images_endpoint(req: PdfToImagesRequest):
    """Render PDF pages as images."""
    import fitz

    path = os.path.abspath(req.path)
    _check_safe(path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    doc = fitz.open(path)
    total = len(doc)

    # Parse pages
    if req.pages:
        page_nums = set()
        for part in req.pages.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                page_nums.update(range(int(a), int(b) + 1))
            else:
                page_nums.add(int(part))
        zero_based = sorted(p - 1 for p in page_nums if 1 <= p <= total)
    else:
        _PAGE_CAP = 50  # prevent OOM on 500-page PDFs (specify pages param for more)
        zero_based = list(range(min(total, _PAGE_CAP)))

    if not zero_based:
        doc.close()
        raise HTTPException(status_code=400, detail=f"No valid pages. PDF has {total} pages.")

    out_folder = req.output_folder or os.path.join(
        os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + "_pages"
    )
    os.makedirs(out_folder, exist_ok=True)

    zoom = req.dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    saved = []
    for page_no in zero_based:
        page = doc[page_no]
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(out_folder, f"page_{page_no + 1}.png")
        pix.save(img_path)
        saved.append(img_path)

    doc.close()
    return {"success": True, "images": saved, "count": len(saved), "output_folder": out_folder}


class ImagesToPdfRequest(BaseModel):
    paths: list
    output_path: str


@router.post("/images-to-pdf")
def images_to_pdf_endpoint(req: ImagesToPdfRequest):
    """Combine images into a single PDF."""
    from PIL import Image

    if not req.paths:
        raise HTTPException(status_code=400, detail="No image paths provided")

    output = os.path.abspath(req.output_path)
    _check_safe(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    images = []
    for p in req.paths:
        p = os.path.abspath(p)
        _check_safe(p)
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail=f"Image not found: {p}")
        img = Image.open(p)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        images.append(img)

    if not images:
        raise HTTPException(status_code=400, detail="No valid images loaded")

    images[0].save(output, "PDF", save_all=True, append_images=images[1:])
    for img in images:
        img.close()

    return {"success": True, "path": output, "pages": len(images)}


# --- Image Tools (Segment 15B) ---


class ResizeImageRequest(BaseModel):
    path: str
    width: int | None = None
    height: int | None = None
    quality: int = 85
    output_path: str | None = None


@router.post("/resize-image")
def resize_image_endpoint(req: ResizeImageRequest):
    """Resize or compress an image."""
    from PIL import Image

    path = os.path.abspath(req.path)
    _check_safe(path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    img = Image.open(path)
    orig_size = img.size

    if req.width and req.height:
        img = img.resize((req.width, req.height), Image.LANCZOS)
    elif req.width:
        ratio = req.width / img.width
        img = img.resize((req.width, int(img.height * ratio)), Image.LANCZOS)
    elif req.height:
        ratio = req.height / img.height
        img = img.resize((int(img.width * ratio), req.height), Image.LANCZOS)

    output = os.path.abspath(req.output_path) if req.output_path else path
    _check_safe(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if img.mode == "RGBA" and output.lower().endswith((".jpg", ".jpeg")):
        img = img.convert("RGB")
    img.save(output, quality=req.quality)
    return {
        "success": True,
        "path": output,
        "original_size": list(orig_size),
        "new_size": list(img.size),
        "file_size": os.path.getsize(output),
    }


class ConvertImageRequest(BaseModel):
    path: str
    format: str  # jpg, png, webp, bmp
    output_path: str | None = None
    quality: int = 90


@router.post("/convert-image")
def convert_image_endpoint(req: ConvertImageRequest):
    """Convert image to a different format."""
    from PIL import Image

    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass

    path = os.path.abspath(req.path)
    _check_safe(path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    fmt = req.format.lower().lstrip(".")
    ext_map = {"jpg": ".jpg", "jpeg": ".jpg", "png": ".png", "webp": ".webp", "bmp": ".bmp"}
    if fmt not in ext_map:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}. Use: jpg, png, webp, bmp")

    output = os.path.abspath(req.output_path) if req.output_path else os.path.splitext(path)[0] + ext_map[fmt]
    _check_safe(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    img = Image.open(path)
    if img.mode == "RGBA" and fmt in ("jpg", "jpeg"):
        img = img.convert("RGB")
    img.save(output, quality=req.quality)
    return {"success": True, "path": output, "format": fmt, "size": os.path.getsize(output)}


class CropImageRequest(BaseModel):
    path: str
    x: int
    y: int
    width: int
    height: int
    output_path: str | None = None


@router.post("/crop-image")
def crop_image_endpoint(req: CropImageRequest):
    """Crop an image to specified dimensions."""
    from PIL import Image

    path = os.path.abspath(req.path)
    _check_safe(path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    img = Image.open(path)
    box = (req.x, req.y, req.x + req.width, req.y + req.height)
    cropped = img.crop(box)

    output = os.path.abspath(req.output_path) if req.output_path else path
    _check_safe(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    cropped.save(output)
    return {"success": True, "path": output, "crop_box": list(box), "new_size": list(cropped.size)}


# --- Image Metadata (EXIF) ---


class ImageMetadataRequest(BaseModel):
    path: str | None = None
    folder: str | None = None


def _extract_exif(filepath: str) -> dict:
    """Extract EXIF metadata from a single image file."""
    from PIL import Image
    from PIL.ExifTags import IFD

    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass

    img = Image.open(filepath)
    result = {
        "path": filepath,
        "dimensions": {"width": img.width, "height": img.height},
        "format": img.format or os.path.splitext(filepath)[1].lstrip(".").upper(),
    }

    exif_data = img.getexif()
    if not exif_data:
        img.close()
        result["exif"] = None
        return result

    def _rational(val):
        """Convert IFDRational or tuple to float."""
        if val is None:
            return None
        if hasattr(val, "numerator"):
            return float(val)
        if isinstance(val, tuple) and len(val) == 2:
            return val[0] / val[1] if val[1] else 0
        return float(val)

    exif = {}
    # DateTimeOriginal
    dt = exif_data.get(36867) or exif_data.get(306)  # DateTimeOriginal or DateTime
    if dt:
        exif["date_taken"] = str(dt).replace(":", "-", 2)  # 2025:02:20 → 2025-02-20

    # Camera
    make = (exif_data.get(271) or "").strip()
    model = (exif_data.get(272) or "").strip()
    if model:
        # Avoid "Canon Canon EOS 5D"
        exif["camera"] = model if make and model.startswith(make) else f"{make} {model}".strip()

    # Lens
    ifd_exif = exif_data.get_ifd(IFD.Exif)
    lens = ifd_exif.get(42036) if ifd_exif else None
    if lens:
        exif["lens"] = str(lens)

    # Focal length
    fl = ifd_exif.get(37386) if ifd_exif else None
    if fl is not None:
        v = _rational(fl)
        exif["focal_length"] = f"{v:.0f}mm" if v else None

    # Aperture
    fn = ifd_exif.get(33437) if ifd_exif else None
    if fn is not None:
        v = _rational(fn)
        exif["aperture"] = f"f/{v:.1f}" if v else None

    # Shutter speed
    et = ifd_exif.get(33434) if ifd_exif else None
    if et is not None:
        v = _rational(et)
        if v and v < 1:
            exif["shutter_speed"] = f"1/{int(round(1 / v))}"
        elif v:
            exif["shutter_speed"] = f"{v:.1f}s"

    # ISO
    iso = ifd_exif.get(34855) if ifd_exif else None
    if iso is not None:
        exif["iso"] = int(iso) if not isinstance(iso, tuple) else int(iso[0])

    # GPS
    try:
        gps_ifd = exif_data.get_ifd(IFD.GPSInfo)
        if gps_ifd:

            def _dms_to_dd(dms, ref):
                d, m, s = [_rational(x) for x in dms]
                dd = d + m / 60 + s / 3600
                return -dd if ref in ("S", "W") else dd

            lat_dms = gps_ifd.get(2)
            lat_ref = gps_ifd.get(1)
            lon_dms = gps_ifd.get(4)
            lon_ref = gps_ifd.get(3)
            if lat_dms and lon_dms:
                exif["gps"] = {
                    "lat": round(_dms_to_dd(lat_dms, lat_ref), 6),
                    "lon": round(_dms_to_dd(lon_dms, lon_ref), 6),
                }
    except Exception:
        pass

    # Orientation
    orient = exif_data.get(274)
    if orient:
        exif["orientation"] = int(orient)

    img.close()
    result["exif"] = exif if exif else None
    return result


IMAGE_EXTS_EXIF = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
    ".heic",
    ".heif",
    ".webp",
    ".bmp",
    ".dng",
    ".cr2",
    ".nef",
    ".arw",
}


@router.post("/image-metadata")
def image_metadata_endpoint(req: ImageMetadataRequest):
    """Extract EXIF metadata from photos."""
    if req.path:
        path = os.path.abspath(req.path)
        _check_safe(path)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        try:
            result = _extract_exif(path)
            result["_hint"] = "Metadata extracted. Answer the user's question about this image."
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")

    elif req.folder:
        folder = os.path.abspath(req.folder)
        _check_safe(folder)
        if not os.path.isdir(folder):
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")

        results = {}
        dates = []
        cameras = set()
        has_gps = 0
        total = 0

        for fname in sorted(os.listdir(folder)):
            if total >= 200:
                break
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTS_EXIF:
                continue
            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                meta = _extract_exif(fpath)
                # Strip full path for batch — just filename
                meta["path"] = fname
                results[fname] = meta
                total += 1
                if meta.get("exif"):
                    if meta["exif"].get("date_taken"):
                        dates.append(meta["exif"]["date_taken"][:10])
                    if meta["exif"].get("camera"):
                        cameras.add(meta["exif"]["camera"])
                    if meta["exif"].get("gps"):
                        has_gps += 1
            except Exception:
                continue

        if not results:
            return {"folder": folder, "images_processed": 0, "results": {}, "_hint": "No images found in folder."}

        summary = {
            "date_range": f"{min(dates)} to {max(dates)}" if dates else None,
            "cameras": sorted(cameras) if cameras else [],
            "has_gps": has_gps,
            "total": total,
        }
        return {
            "folder": folder,
            "images_processed": total,
            "results": results,
            "summary": summary,
            "_hint": f"Metadata for {total} images. Summary shows date range and cameras used.",
        }

    else:
        raise HTTPException(status_code=400, detail="Provide path (single image) or folder (batch).")


# --- Archive Tools (Segment 15B) ---


class CompressFilesRequest(BaseModel):
    paths: list[str]
    output_path: str


@router.post("/compress-files")
def compress_files_endpoint(req: CompressFilesRequest):
    """Compress files into a zip archive."""
    import zipfile

    output = os.path.abspath(req.output_path)
    _check_safe(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    added = 0
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in req.paths:
            p = os.path.abspath(p)
            _check_safe(p)
            if not os.path.exists(p):
                continue
            if os.path.isdir(p):
                for root, dirs, files in os.walk(p):
                    for f in files:
                        fp = os.path.join(root, f)
                        arcname = os.path.relpath(fp, os.path.dirname(p))
                        zf.write(fp, arcname)
                        added += 1
            else:
                zf.write(p, os.path.basename(p))
                added += 1

    return {"success": True, "path": output, "files_added": added, "archive_size": os.path.getsize(output)}


class ExtractArchiveRequest(BaseModel):
    path: str
    output_path: str | None = None


@router.post("/extract-archive")
def extract_archive_endpoint(req: ExtractArchiveRequest):
    """Extract a zip archive."""
    import zipfile

    path = os.path.abspath(req.path)
    _check_safe(path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    output = os.path.abspath(req.output_path) if req.output_path else os.path.splitext(path)[0]
    _check_safe(output)

    try:
        with zipfile.ZipFile(path, "r") as zf:
            # Security: check for path traversal (compute output abspath once)
            output_abs = os.path.abspath(output)
            for name in zf.namelist():
                target = os.path.join(output_abs, name)
                # normpath resolves .. without hitting disk (faster than abspath)
                if not os.path.normpath(target).startswith(output_abs):
                    raise HTTPException(status_code=400, detail=f"Unsafe path in archive: {name}")
            zf.extractall(output)
            return {"success": True, "path": output_abs, "files_extracted": len(zf.namelist())}
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Not a valid zip file")


# --- Download (Segment 15B) ---


class DownloadUrlRequest(BaseModel):
    url: str
    save_path: str | None = None


@router.post("/download-url")
def download_url_endpoint(req: DownloadUrlRequest):
    """Download a file from a URL."""
    import urllib.error
    import urllib.request

    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    # Determine save path
    if req.save_path:
        save_path = os.path.abspath(req.save_path)
    else:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        filename = os.path.basename(parsed.path) or "downloaded_file"
        save_dir = os.path.join(os.path.expanduser("~"), "Downloads", "Pinpoint")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

    _check_safe(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        req_obj = urllib.request.Request(url, headers={"User-Agent": "Pinpoint/1.0"})
        with urllib.request.urlopen(req_obj, timeout=60) as resp:
            with open(save_path, "wb") as f:
                shutil.copyfileobj(resp, f)  # stream in chunks, not buffer entire file in RAM
        return {
            "success": True,
            "path": save_path,
            "size": os.path.getsize(save_path),
            "url": url,
            "_hint": "Use index_file to make this file searchable, or send_file to share it.",
        }
    except urllib.error.URLError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")


# --- Run Python (Segment 15D) ---

PYTHON_WORK_DIR = "/tmp/pinpoint_python"
os.makedirs(PYTHON_WORK_DIR, exist_ok=True)


class RunPythonRequest(BaseModel):
    code: str
    timeout: int = 30


@router.post("/run-python")
def run_python_endpoint(req: RunPythonRequest):
    """Execute Python code and return stdout + created files."""
    import contextlib
    import io
    import signal

    timeout = min(req.timeout, 120)

    # Snapshot files before execution
    before = set()
    for root, dirs, files in os.walk(PYTHON_WORK_DIR):
        for f in files:
            before.add(os.path.join(root, f))

    # Pre-loaded namespace
    namespace = {
        "__builtins__": __builtins__,
        "WORK_DIR": PYTHON_WORK_DIR,
    }

    # Lazy imports inside namespace
    setup_code = f"""
import os, sys, json, math, re, pathlib, shutil, glob, hashlib, datetime, io, csv
WORK_DIR = "{PYTHON_WORK_DIR}"
os.chdir(WORK_DIR)
try:
    import numpy as np
except ImportError:
    pass
try:
    import pandas as pd
except ImportError:
    pass
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    pass
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    pass
"""

    full_code = setup_code + "\n" + req.code

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution timed out after {timeout}s")

    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(full_code, namespace)

        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    except TimeoutError as e:
        signal.alarm(0)
        return {"success": False, "error": str(e), "stdout": stdout_capture.getvalue()[:5000]}
    except Exception as e:
        signal.alarm(0)
        return {"success": False, "error": f"{type(e).__name__}: {e}", "stdout": stdout_capture.getvalue()[:5000]}

    # Find new/modified files
    after = set()
    for root, dirs, files in os.walk(PYTHON_WORK_DIR):
        for f in files:
            after.add(os.path.join(root, f))
    new_files = sorted(after - before)

    stdout_text = stdout_capture.getvalue()[:10000]
    stderr_text = stderr_capture.getvalue()[:2000]

    result = {"success": True, "stdout": stdout_text}
    if stderr_text:
        result["stderr"] = stderr_text
    if new_files:
        result["files_created"] = new_files[:50]  # cap to prevent huge JSON responses
        if len(new_files) > 50:
            result["files_created_total"] = len(new_files)
    return result
