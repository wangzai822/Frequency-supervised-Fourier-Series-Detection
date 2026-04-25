# -*- coding: utf-8 -*-
"""
FS-FSD WebGL Damage Intelligence Viewer - FastAPI Server
========================================================

This server is a compatibility layer between the frontend app.js and the
real project service layer:

    webgl_viewer/fsd_service.py
    webgl_viewer/state.py

Important:
- Do not bypass fsd_service.py.
- fsd_service.py already implements:
  1. schema auto-detection
  2. folder-driven image catalog
  3. image/defect matching
  4. Fourier coefficient decoding
  5. polygon reconstruction
  6. local image file resolving

This server only provides stable API endpoints for the frontend.
"""

from __future__ import annotations

import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .fsd_service import (
    FSDServiceError,
    get_defect,
    get_image_file_path,
    get_image_records,
    get_runtime_state,
    get_schema,
    get_summary,
    list_images,
    open_archive,
)
from .state import VIEWER_STATE


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(
    title="FS-FSD WebGL Damage Intelligence Viewer",
    description="Local viewer for Fourier-based structural damage databases",
    version="0.5.1-service",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _slash(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _error_response(exc: Exception, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=str(exc))


def _normalize_bool(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on", "y"}


def _basename(value: Any) -> str:
    text = _slash(value)
    if not text:
        return ""
    return text.split("/")[-1]


def _frontend_summary_aliases(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert fsd_service summary fields into frontend-friendly aliases.

    fsd_service.py usually uses:
        image_count
        defect_count
        class_count
        db_file_size
        image_found_count
        missing_image_count

    The frontend accepts several aliases, but returning common aliases keeps
    old/new app.js versions stable.
    """
    src = dict(summary or {})

    image_count = src.get("image_count", src.get("images", 0))
    defect_count = src.get("defect_count", src.get("defects", 0))
    class_count = src.get("class_count", src.get("classes", 0))
    db_file_size = src.get("db_file_size", src.get("db_size", None))
    image_found_count = src.get("image_found_count", src.get("images_found", 0))
    missing_image_count = src.get("missing_image_count", src.get("missing_images", 0))

    src.update(
        {
            "images": image_count,
            "image_count": image_count,
            "defects": defect_count,
            "defect_count": defect_count,
            "classes": class_count,
            "class_count": class_count,
            "db_size": db_file_size,
            "db_file_size": db_file_size,
            "images_found": image_found_count,
            "image_found_count": image_found_count,
            "missing_images": missing_image_count,
            "missing_image_count": missing_image_count,
            "image_root": src.get("image_dir") or src.get("image_root"),
            "image_repository": src.get("image_dir") or src.get("image_root"),
        }
    )

    return src


def _frontend_image_aliases(image: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """
    Convert fsd_service image fields into frontend-friendly image fields.

    fsd_service.py returns image entries like:
        image_key
        image_id
        source_path
        resolved_path
        image_exists
        det_count
        img_w
        img_h

    app.js also understands:
        key
        name
        file
        path
        exists
        defectCount
        width
        height
    """
    src = dict(image or {})

    image_key = (
        src.get("image_key")
        or src.get("key")
        or src.get("id")
        or src.get("image_id")
        or f"image-{index + 1}"
    )

    source_path = src.get("source_path") or src.get("path") or src.get("file") or ""
    resolved_path = src.get("resolved_path") or ""
    image_id = src.get("image_id") or src.get("id") or ""

    filename = (
        src.get("filename")
        or src.get("file_name")
        or src.get("name")
        or _basename(source_path)
        or _basename(resolved_path)
        or str(image_id)
        or str(image_key)
    )

    exists = src.get("image_exists")
    if exists is None:
        exists = src.get("exists")
    if exists is None:
        exists = bool(resolved_path)

    det_count = src.get("det_count")
    if det_count is None:
        det_count = src.get("defectCount")
    if det_count is None:
        det_count = src.get("defect_count")
    if det_count is None:
        det_count = src.get("damage_count")
    if det_count is None:
        det_count = 0

    width = src.get("img_w", src.get("width", src.get("image_width")))
    height = src.get("img_h", src.get("height", src.get("image_height")))

    src.update(
        {
            "__index": index,

            # frontend canonical fields
            "key": str(image_key),
            "id": image_id or image_key,
            "name": filename,
            "file": source_path or resolved_path,
            "path": source_path or resolved_path,
            "exists": bool(exists),
            "defectCount": det_count,
            "width": width,
            "height": height,

            # service canonical fields
            "image_key": str(image_key),
            "image_id": image_id or filename,
            "source_path": source_path,
            "resolved_path": resolved_path,
            "image_exists": bool(exists),
            "det_count": det_count,
            "img_w": width,
            "img_h": height,
        }
    )

    return src


def _frontend_record_aliases(record: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """
    Convert one defect record into fields app.js can consume.
    fsd_service.get_image_records() should already include reconstructed polygon.
    """
    src = dict(record or {})
    src.setdefault("__index", index)

    if "class" not in src and "class_name" in src:
        src["class"] = src["class_name"]

    if "damage_type" not in src and "class_name" in src:
        src["damage_type"] = src["class_name"]

    if "polygon_points" not in src and "polygon" in src:
        src["polygon_points"] = src["polygon"]

    if "polygon" not in src and "polygon_points" in src:
        src["polygon"] = src["polygon_points"]

    if "area" not in src and "area_px2" in src:
        src["area"] = src["area_px2"]

    if "perimeter" not in src and "perimeter_px" in src:
        src["perimeter"] = src["perimeter_px"]

    if "orientation" not in src and "orientation_deg" in src:
        src["orientation"] = src["orientation_deg"]

    return src


def _list_frontend_images(
    *,
    query: str = "",
    only_existing: bool = False,
    limit: int = 50000,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    list_images() wrapper that returns frontend-compatible image objects.
    """
    result = list_images(
        query=query or None,
        only_existing=only_existing,
        limit=limit,
        offset=offset,
    )

    raw_images = result.get("images", []) or []
    images = [
        _frontend_image_aliases(img, index=offset + i)
        for i, img in enumerate(raw_images)
    ]

    return {
        **result,
        "images": images,
    }


def _find_image_key_by_loose_query(
    *,
    key: str = "",
    image_id: str = "",
    path: str = "",
    name: str = "",
) -> Optional[str]:
    """
    If frontend sends image_id/path/name instead of image_key, locate the real
    image_key from VIEWER_STATE or from fsd_service.list_images().
    """
    candidates = [
        key,
        image_id,
        path,
        name,
        _basename(path),
        Path(_slash(path)).stem if path else "",
        _basename(name),
        Path(_slash(name)).stem if name else "",
    ]

    norm_candidates = {
        str(x).strip().lower().replace("\\", "/")
        for x in candidates
        if str(x or "").strip()
    }

    if not norm_candidates:
        return None

    raw_images: Any = []

    try:
        raw_images = VIEWER_STATE.list_images()
    except Exception:
        try:
            raw_images = list_images(
                query=None,
                only_existing=False,
                limit=50000,
                offset=0,
            )
        except Exception:
            raw_images = []

    if isinstance(raw_images, dict):
        images = raw_images.get("images", []) or []
    else:
        images = raw_images or []

    for img in images:
        if not isinstance(img, dict):
            continue

        img_values = [
            img.get("image_key"),
            img.get("key"),
            img.get("image_id"),
            img.get("id"),
            img.get("source_path"),
            img.get("resolved_path"),
            img.get("file"),
            img.get("path"),
            img.get("name"),
            _basename(img.get("source_path")),
            Path(_slash(img.get("source_path"))).stem if img.get("source_path") else "",
            _basename(img.get("resolved_path")),
            Path(_slash(img.get("resolved_path"))).stem if img.get("resolved_path") else "",
            _basename(img.get("path")),
            Path(_slash(img.get("path"))).stem if img.get("path") else "",
        ]

        img_norm = {
            str(x).strip().lower().replace("\\", "/")
            for x in img_values
            if str(x or "").strip()
        }

        if norm_candidates & img_norm:
            return str(img.get("image_key") or img.get("key") or img.get("id"))

    return None


# -----------------------------------------------------------------------------
# Native file dialogs
# -----------------------------------------------------------------------------

def _open_file_dialog(title: str, filetypes: Sequence[Tuple[str, str]]) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"[select-db] tkinter unavailable: {exc}")
        return ""

    root = tk.Tk()
    root.withdraw()

    try:
        root.attributes("-topmost", True)
        root.update()
    except Exception:
        pass

    try:
        selected = filedialog.askopenfilename(
            parent=root,
            title=title,
            initialdir=str(Path.cwd()),
            filetypes=list(filetypes),
        )
        return selected or ""
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def _open_directory_dialog(title: str) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"[select-folder] tkinter unavailable: {exc}")
        return ""

    root = tk.Tk()
    root.withdraw()

    try:
        root.attributes("-topmost", True)
        root.update()
    except Exception:
        pass

    try:
        selected = filedialog.askdirectory(
            parent=root,
            title=title,
            initialdir=str(Path.cwd()),
            mustexist=True,
        )
        return selected or ""
    finally:
        try:
            root.destroy()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Page route
# -----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """
    Serve static/index.html first because this project commonly places index.html
    under webgl_viewer/static/.
    """
    for path in [
        STATIC_DIR / "index.html",
        TEMPLATE_DIR / "index.html",
    ]:
        if path.exists():
            return HTMLResponse(path.read_text(encoding="utf-8"))

    return HTMLResponse(
        "<h1>FS-FSD WebGL Viewer</h1><p>index.html not found.</p>",
        status_code=404,
    )


# -----------------------------------------------------------------------------
# Health / diagnostics
# -----------------------------------------------------------------------------

@app.get("/api/health")
@app.get("/api/ping")
def api_health() -> Dict[str, Any]:
    snapshot = get_runtime_state()

    return {
        "ok": True,
        "service": "FS-FSD WebGL Damage Intelligence Viewer",
        "version": "0.5.1-service",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": os.sys.platform,
        "ready": snapshot.get("ready", False),
        "db_loaded": snapshot.get("ready", False),
        "db_path": snapshot.get("db_path"),
        "image_root": snapshot.get("image_dir"),
        "image_dir": snapshot.get("image_dir"),
        "images": snapshot.get("image_count", 0),
        "defects": snapshot.get("defect_count", 0),
        "last_error": snapshot.get("last_error"),
    }


@app.get("/api/state")
@app.get("/api/runtime-state")
def api_state() -> Dict[str, Any]:
    return {
        "ok": True,
        "state": get_runtime_state(),
    }


@app.get("/api/schema")
@app.get("/api/archive/schema")
def api_schema() -> Dict[str, Any]:
    try:
        return {
            "ok": True,
            "schema": get_schema(),
        }
    except Exception as exc:
        raise _error_response(exc)


# -----------------------------------------------------------------------------
# Native pickers
# -----------------------------------------------------------------------------

@app.post("/api/select-db")
@app.get("/api/select-db")
@app.post("/api/select_db")
@app.get("/api/select_db")
def api_select_db() -> Dict[str, Any]:
    path = _open_file_dialog(
        "Select SQLite damage archive",
        [
            ("SQLite database", "*.db *.sqlite *.sqlite3"),
            ("All files", "*.*"),
        ],
    )
    path = _slash(path)

    return {
        "ok": bool(path),
        "cancelled": not bool(path),
        "path": path,
        "db_path": path,
        "database_path": path,
        "sqlite_path": path,
    }


@app.post("/api/select-folder")
@app.get("/api/select-folder")
@app.post("/api/select_folder")
@app.get("/api/select_folder")
@app.post("/api/pick-folder")
@app.get("/api/pick-folder")
@app.post("/api/pick_folder")
@app.get("/api/pick_folder")
@app.post("/api/select-directory")
@app.get("/api/select-directory")
@app.post("/api/select-dir")
@app.get("/api/select-dir")
def api_select_folder() -> Dict[str, Any]:
    folder = _open_directory_dialog("Select image repository")
    folder = _slash(folder)

    return {
        "ok": bool(folder),
        "cancelled": not bool(folder),
        "path": folder,
        "folder": folder,
        "directory": folder,
        "image_root": folder,
        "image_folder": folder,
        "image_repository": folder,
    }


# -----------------------------------------------------------------------------
# Archive open / initialize
# -----------------------------------------------------------------------------

@app.post("/api/initialize")
@app.post("/api/init")
@app.post("/api/archive/init")
@app.post("/api/archive/initialize")
@app.post("/api/load_archive")
@app.post("/api/archive")
@app.post("/api/open-archive")
def api_initialize(payload: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
    payload = payload or {}

    db_path = (
        payload.get("db_path")
        or payload.get("database_path")
        or payload.get("sqlite_path")
        or payload.get("dbPath")
        or payload.get("database")
        or ""
    )

    image_root = (
        payload.get("image_root")
        or payload.get("image_folder")
        or payload.get("image_repository")
        or payload.get("imageRoot")
        or payload.get("root")
        or ""
    )

    db_path = _slash(db_path)
    image_root = _slash(image_root)

    if not db_path:
        raise HTTPException(status_code=400, detail="db_path is required.")

    try:
        summary = open_archive(db_path=db_path, image_dir=image_root)
        summary = _frontend_summary_aliases(summary)

        image_result = _list_frontend_images(
            query="",
            only_existing=False,
            limit=50000,
            offset=0,
        )

        return {
            "ok": True,
            "db_path": summary.get("db_path") or db_path,
            "image_root": summary.get("image_dir") or image_root,
            "image_dir": summary.get("image_dir") or image_root,
            "summary": summary,
            "total": image_result.get("total", 0),
            "images": image_result.get("images", []),
        }

    except FSDServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to initialize archive: {exc}")


# -----------------------------------------------------------------------------
# Summary / image list
# -----------------------------------------------------------------------------

@app.get("/api/summary")
@app.get("/api/archive/summary")
def api_summary() -> Dict[str, Any]:
    try:
        summary = _frontend_summary_aliases(get_summary())
        return {
            "ok": True,
            "summary": summary,
            "db_path": summary.get("db_path"),
            "image_root": summary.get("image_dir"),
            "image_dir": summary.get("image_dir"),
        }
    except Exception as exc:
        raise _error_response(exc)


@app.get("/api/images")
@app.get("/api/archive/images")
@app.get("/api/image_records")
@app.get("/api/image-records")
@app.get("/api/list_images")
@app.get("/api/list-images")
def api_images(
    search: str = Query(default=""),
    q: str = Query(default=""),
    keyword: str = Query(default=""),
    existing_only: str = Query(default="0"),
    only_existing: str = Query(default="0"),
    existing: str = Query(default="0"),
    page: int = Query(default=1),
    page_size: int = Query(default=5000),
    limit: int = Query(default=5000),
) -> Dict[str, Any]:
    query = (search or q or keyword or "").strip()

    only = (
        _normalize_bool(existing_only)
        or _normalize_bool(only_existing)
        or _normalize_bool(existing)
    )

    page = max(1, int(page or 1))
    size = max(1, min(int(page_size or limit or 5000), 50000))
    offset = (page - 1) * size

    try:
        result = _list_frontend_images(
            query=query,
            only_existing=only,
            limit=size,
            offset=offset,
        )
        summary = _frontend_summary_aliases(get_summary())

        return {
            "ok": True,
            "summary": summary,
            "total": result.get("total", 0),
            "page": page,
            "page_size": size,
            "images": result.get("images", []),
        }

    except Exception as exc:
        raise _error_response(exc)


# -----------------------------------------------------------------------------
# Records / defects / polygons
# -----------------------------------------------------------------------------

@app.get("/api/records")
@app.get("/api/damage_records")
@app.get("/api/damage-records")
@app.get("/api/defects")
@app.get("/api/annotations")
@app.get("/api/image/records")
@app.get("/api/image_records/records")
def api_records(
    key: str = Query(default=""),
    image_key: str = Query(default=""),
    image_id: str = Query(default=""),
    id: str = Query(default=""),
    path: str = Query(default=""),
    name: str = Query(default=""),
    polygon_points: int = Query(default=256),
    points: int = Query(default=0),
    limit: int = Query(default=20000),
) -> Dict[str, Any]:
    """
    Return records for one image.

    Frontend normally sends:
        key=<image_key>

    This route also supports loose matching by image_id/path/name.
    """
    point_count = int(points or polygon_points or 256)
    key_value = key or image_key or id or image_id

    resolved_key = None
    data = None

    if key_value:
        # First try directly as image_key.
        try:
            data = get_image_records(
                str(key_value),
                polygon_points=point_count,
            )
            resolved_key = str(key_value)
        except Exception:
            data = None

    if data is None:
        resolved_key = _find_image_key_by_loose_query(
            key=key_value,
            image_id=image_id,
            path=path,
            name=name,
        )

        if not resolved_key:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Image not found for records request. "
                    f"key={key_value}, image_id={image_id}, path={path}, name={name}"
                ),
            )

        try:
            data = get_image_records(
                resolved_key,
                polygon_points=point_count,
            )
        except FSDServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    raw_records = data.get("records", []) or []
    max_limit = max(1, int(limit or 20000))

    records = [
        _frontend_record_aliases(record, index=i)
        for i, record in enumerate(raw_records[:max_limit])
    ]

    image = _frontend_image_aliases(data.get("image", {}), index=0)

    return {
        "ok": True,
        "image_key": resolved_key or key_value,
        "image": image,
        "total": len(records),
        "record_count": len(records),
        "records": records,
    }


@app.get("/api/images/{image_key}/records")
@app.get("/api/image/{image_key}/records")
@app.get("/api/records/{image_key}")
def api_records_by_path(
    image_key: str,
    polygon_points: int = Query(default=256),
) -> Dict[str, Any]:
    try:
        data = get_image_records(
            image_key,
            polygon_points=int(polygon_points or 256),
        )

        records = [
            _frontend_record_aliases(record, index=i)
            for i, record in enumerate(data.get("records", []) or [])
        ]

        image = _frontend_image_aliases(data.get("image", {}), index=0)

        return {
            "ok": True,
            "image_key": image_key,
            "image": image,
            "total": len(records),
            "record_count": len(records),
            "records": records,
        }

    except FSDServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/defect/{defect_id}")
@app.get("/api/record/{defect_id}")
def api_defect(
    defect_id: str,
    polygon_points: int = Query(default=256),
    include_coefficients: str = Query(default="1"),
) -> Dict[str, Any]:
    try:
        record = get_defect(
            defect_id,
            include_polygon=True,
            include_coefficients=_normalize_bool(include_coefficients),
            polygon_points=int(polygon_points or 256),
        )

        return {
            "ok": True,
            "record": _frontend_record_aliases(record, index=0),
        }

    except FSDServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------------
# Image file
# -----------------------------------------------------------------------------

@app.get("/api/image")
@app.get("/api/image_file")
@app.get("/api/image-file")
@app.get("/api/image/data")
@app.get("/api/image_data")
@app.get("/api/serve_image")
@app.get("/image")
def api_image(
    key: str = Query(default=""),
    image_key: str = Query(default=""),
    image_id: str = Query(default=""),
    id: str = Query(default=""),
    path: str = Query(default=""),
    file: str = Query(default=""),
    filename: str = Query(default=""),
    name: str = Query(default=""),
):
    key_value = key or image_key or id or image_id

    if key_value:
        try:
            p = Path(get_image_file_path(str(key_value)))
            return FileResponse(str(p), media_type=_guess_mime(p))
        except Exception:
            pass

    resolved_key = _find_image_key_by_loose_query(
        key=key_value,
        image_id=image_id,
        path=path or file or filename,
        name=name,
    )

    if resolved_key:
        try:
            p = Path(get_image_file_path(str(resolved_key)))
            return FileResponse(str(p), media_type=_guess_mime(p))
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    raise HTTPException(
        status_code=404,
        detail=(
            "Image file not found. "
            f"key={key_value}, image_id={image_id}, path={path or file or filename}, name={name}"
        ),
    )