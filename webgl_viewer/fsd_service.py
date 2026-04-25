# -*- coding: utf-8 -*-
"""
FS-FSD WebGL Damage Intelligence Viewer - Database Service
==========================================================

This module is the backend service layer for the WebGL viewer.

Responsibilities
----------------
1. Open an FS-FSD SQLite database.
2. Inspect schema and table columns.
3. Auto-detect image, defect, and label tables from flexible schemas.
4. Load image records, defect records, and label map.
5. Decode Fourier coefficient BLOBs / JSON / arrays.
6. Reconstruct damage polygons through `fsd_geometry.py` when available.
7. Resolve local image file paths.
8. Provide clean JSON-ready data for FastAPI routes.

Important display behavior
--------------------------
If the user selects an Image Repository folder, the frontend image browser is
driven by that selected folder only.

That means:
- Visible images = image files inside the selected folder.
- Database image rows outside the selected folder are not displayed.
- Database defects are attached only if they match images in the selected folder.
- Folder images without matched defects are still displayed with det_count = 0.
- SQLite is still used for defect records, Fourier coefficients and polygons.

Why this version exists
-----------------------
Your current archive path:

    runs/archive/fsd_val377/fsd_archive.db

does not use the previous simple table names such as:

    image_records
    defect_records
    label_map

Therefore this service now performs schema scoring and table auto-detection
instead of relying only on exact table names.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import numpy as np

from .state import VIEWER_STATE


# ---------------------------------------------------------------------
# Project path configuration
# ---------------------------------------------------------------------
FILE = Path(__file__).resolve()
WEBGL_DIR = FILE.parent
SQLITE_DIR = WEBGL_DIR.parent
PROJECT_DIR = SQLITE_DIR.parent

for p in [SQLITE_DIR, PROJECT_DIR]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


# ---------------------------------------------------------------------
# Optional geometry backend from your existing project
# ---------------------------------------------------------------------
try:
    from fsd_geometry import reconstruct_polygon_from_coeffs as _fsd_reconstruct_polygon
except Exception:
    _fsd_reconstruct_polygon = None


# ---------------------------------------------------------------------
# Table and column candidates
# ---------------------------------------------------------------------
IMAGE_TABLE_CANDIDATES = [
    "image_records",
    "images",
    "image_record",
    "image_table",
    "image_index",
    "image_metadata",
    "archive_images",
    "source_images",
    "files",
    "file_records",
    "image_files",
    "dataset_images",
]

DEFECT_TABLE_CANDIDATES = [
    "defect_records",
    "defects",
    "damage_records",
    "annotations",
    "annotation_records",
    "detections",
    "detection_records",
    "predictions",
    "prediction_records",
    "instances",
    "instance_records",
    "objects",
    "object_records",
    "segments",
    "segmentations",
    "masks",
    "mask_records",
    "damage_instances",
    "defect_instances",
    "fsd_records",
    "fsd_record",
    "fsd_defects",
    "fsd_annotations",
    "fsd_instances",
    "fs_fsd_records",
    "fs_fsd_defects",
    "fourier_records",
    "fourier_defects",
    "archive_records",
    "records",
]

LABEL_TABLE_CANDIDATES = [
    "label_map",
    "labels",
    "classes",
    "class_map",
    "categories",
    "category_map",
    "label_records",
    "class_records",
    "category_records",
    "names",
]

IMAGE_TABLE_NAME_TOKENS = [
    "image",
    "img",
    "file",
    "source",
    "frame",
]

DEFECT_TABLE_NAME_TOKENS = [
    "defect",
    "damage",
    "annotation",
    "annot",
    "detection",
    "detect",
    "prediction",
    "pred",
    "instance",
    "object",
    "mask",
    "segment",
    "fsd",
    "fourier",
    "record",
]

LABEL_TABLE_NAME_TOKENS = [
    "label",
    "class",
    "category",
    "name",
]

IMAGE_LINK_COLUMNS = [
    "image_pk",
    "image_fk",
    "img_pk",
    "img_fk",
    "image_uid",
    "img_uid",
    "image_uuid",
    "image_id",
    "img_id",
    "file_id",
    "frame_id",
    "image_name",
    "img_name",
    "filename",
    "file_name",
    "image_file",
    "source_path",
    "image_path",
    "file_path",
    "path",
    "relative_path",
]

IMAGE_PATH_COLUMNS = [
    "source_path",
    "image_path",
    "file_path",
    "path",
    "relative_path",
    "filename",
    "file_name",
    "image_file",
    "image_name",
]

IMAGE_SIZE_COLUMNS = [
    "img_w",
    "img_h",
    "image_width",
    "image_height",
    "width",
    "height",
    "w",
    "h",
]

CLASS_COLUMNS = [
    "class_id",
    "cls_id",
    "category_id",
    "label_id",
    "class_name",
    "category",
    "label",
    "name",
    "damage_type",
    "defect_type",
]

DEFECT_ID_COLUMNS = [
    "defect_id",
    "defect_pk",
    "damage_id",
    "annotation_id",
    "instance_id",
    "object_id",
    "detection_id",
    "id",
    "pk",
]

GEOMETRY_COLUMNS = [
    "centroid_x_px",
    "centroid_y_px",
    "cx",
    "cy",
    "center_x",
    "center_y",
    "x_center",
    "y_center",
    "area_px2",
    "area",
    "mask_area",
    "perimeter_px",
    "perimeter",
    "orientation_deg",
    "orientation",
    "angle_deg",
    "elongation",
    "aspect_ratio",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "x1",
    "y1",
    "x2",
    "y2",
]

SCORE_COLUMNS = [
    "score",
    "confidence",
    "conf",
    "probability",
    "prob",
]

FOURIER_COLUMNS = [
    "fourier_coeffs",
    "fourier_coefficients",
    "fsd_coeffs",
    "fsd_coefficients",
    "coeffs",
    "coefficients",
    "descriptor",
    "fourier_descriptor",
    "fourier_blob",
    "coeff_blob",
    "coeff_data",
    "coefficient_blob",
    "descriptor_blob",
    "fsd_blob",
    "fsd_data",
    "fsd_descriptor",
    "shape_coeffs",
    "shape_coefficients",
    "shape_descriptor",
    "contour_coeffs",
    "contour_coefficients",
    "contour_descriptor",
    "efd",
    "efd_coeffs",
    "efd_coefficients",
    "elliptic_fourier_descriptor",
    "fourier_f16",
    "fourier_fp16",
    "coeffs_f16",
    "coefficients_f16",
]

FOURIER_FUZZY_TOKENS = [
    "fourier",
    "coeff",
    "descriptor",
    "fsd",
    "efd",
    "harmonic",
    "contour",
    "shape",
]

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
]


# ---------------------------------------------------------------------
# Service exception
# ---------------------------------------------------------------------
class FSDServiceError(RuntimeError):
    """
    User-facing service exception for API routes.
    """


# ---------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------
@contextmanager
def sqlite_connection(db_path: Path):
    """
    Create a short-lived SQLite connection.

    We intentionally do not keep a persistent sqlite3.Connection in global state
    because FastAPI/Uvicorn can access it from different threads.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        yield conn
    finally:
        conn.close()


def quote_identifier(name: str) -> str:
    """
    Safely quote a SQLite identifier.
    """
    return '"' + str(name).replace('"', '""') + '"'


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """
    Convert sqlite3.Row to plain dict.
    """
    return {k: row[k] for k in row.keys()}


def read_all_rows(conn: sqlite3.Connection, table: Optional[str]) -> List[Dict[str, Any]]:
    """
    Read all rows from a table.

    Returns an empty list if table is None.
    """
    if not table:
        return []

    rows = conn.execute(f"SELECT * FROM {quote_identifier(table)}").fetchall()
    return [row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------
# Schema auto-detection helpers
# ---------------------------------------------------------------------
def choose_table(tables: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """
    Choose the first existing table from candidates, case-insensitively.
    """
    lower_to_real = {t.lower(): t for t in tables}

    for c in candidates:
        if c.lower() in lower_to_real:
            return lower_to_real[c.lower()]

    return None


def schema_column_names(schema: Dict[str, Any], table: str) -> List[str]:
    """
    Return lowercase column names for a table.
    """
    columns = schema.get("columns", {}).get(table, [])
    return [str(c.get("name", "")).lower() for c in columns]


def schema_column_types(schema: Dict[str, Any], table: str) -> Dict[str, str]:
    """
    Return lowercase column name -> lowercase SQLite type.
    """
    out: Dict[str, str] = {}
    columns = schema.get("columns", {}).get(table, [])

    for c in columns:
        name = str(c.get("name", "")).lower()
        typ = str(c.get("type", "") or "").lower()
        out[name] = typ

    return out


def table_row_count(schema: Dict[str, Any], table: str) -> int:
    """
    Return row count for a table.
    """
    try:
        value = schema.get("row_counts", {}).get(table, 0)
        return int(value or 0)
    except Exception:
        return 0


def has_any_column(cols: Sequence[str], names: Sequence[str]) -> bool:
    """
    True if any exact lowercase column name exists.
    """
    colset = {str(c).lower() for c in cols}
    return any(str(n).lower() in colset for n in names)


def count_any_column(cols: Sequence[str], names: Sequence[str]) -> int:
    """
    Count exact lowercase column name hits.
    """
    colset = {str(c).lower() for c in cols}
    return sum(1 for n in names if str(n).lower() in colset)


def has_token_column(cols: Sequence[str], tokens: Sequence[str]) -> bool:
    """
    True if any column contains any token.
    """
    for col in cols:
        lc = str(col).lower()
        for tok in tokens:
            if str(tok).lower() in lc:
                return True
    return False


def count_token_column(cols: Sequence[str], tokens: Sequence[str]) -> int:
    """
    Count token-based column hits.
    """
    count = 0

    for col in cols:
        lc = str(col).lower()
        if any(str(tok).lower() in lc for tok in tokens):
            count += 1

    return count


def table_name_bonus(table: str, candidates: Sequence[str], tokens: Sequence[str]) -> float:
    """
    Score table name by exact candidate and fuzzy token matches.
    """
    name = str(table).lower()

    for candidate in candidates:
        c = str(candidate).lower()
        if name == c:
            return 90.0

    for candidate in candidates:
        c = str(candidate).lower()
        if name.endswith(c) or name.startswith(c):
            return 45.0

    bonus = 0.0
    for token in tokens:
        if str(token).lower() in name:
            bonus += 8.0

    return min(bonus, 32.0)


def has_fourier_column_by_schema(schema: Dict[str, Any], table: str) -> bool:
    """
    True if schema suggests a Fourier/descriptor/coefficients column.
    """
    cols = schema_column_names(schema, table)
    return has_any_column(cols, FOURIER_COLUMNS) or has_token_column(cols, FOURIER_FUZZY_TOKENS)


def score_image_table(schema: Dict[str, Any], table: str) -> float:
    """
    Score how likely a table is an image table.
    """
    cols = schema_column_names(schema, table)
    score = 0.0

    score += table_name_bonus(table, IMAGE_TABLE_CANDIDATES, IMAGE_TABLE_NAME_TOKENS)
    score += count_any_column(cols, IMAGE_LINK_COLUMNS) * 3.0
    score += count_any_column(cols, IMAGE_PATH_COLUMNS) * 7.0
    score += count_any_column(cols, IMAGE_SIZE_COLUMNS) * 4.0

    if has_fourier_column_by_schema(schema, table):
        score -= 14.0

    if has_any_column(cols, CLASS_COLUMNS) and not has_any_column(cols, IMAGE_PATH_COLUMNS):
        score -= 4.0

    rows = table_row_count(schema, table)
    if rows > 0:
        score += min(4.0, math.log10(rows + 1.0))

    return score


def score_label_table(schema: Dict[str, Any], table: str) -> float:
    """
    Score how likely a table is a label/category/class map table.
    """
    cols = schema_column_names(schema, table)
    score = 0.0

    score += table_name_bonus(table, LABEL_TABLE_CANDIDATES, LABEL_TABLE_NAME_TOKENS)

    id_hits = count_any_column(cols, ["class_id", "id", "label_id", "category_id", "cls_id"])
    name_hits = count_any_column(cols, ["class_name", "name", "label", "category"])

    score += id_hits * 8.0
    score += name_hits * 8.0

    if has_fourier_column_by_schema(schema, table):
        score -= 20.0

    if has_any_column(cols, IMAGE_PATH_COLUMNS):
        score -= 8.0

    rows = table_row_count(schema, table)
    if 0 < rows <= 1000:
        score += 3.0

    return score


def score_defect_table(schema: Dict[str, Any], table: str) -> float:
    """
    Score how likely a table is a defect/damage/annotation table.
    """
    cols = schema_column_names(schema, table)
    types = schema_column_types(schema, table)

    score = 0.0

    score += table_name_bonus(table, DEFECT_TABLE_CANDIDATES, DEFECT_TABLE_NAME_TOKENS)

    exact_fourier_hits = count_any_column(cols, FOURIER_COLUMNS)
    fuzzy_fourier_hits = count_token_column(cols, FOURIER_FUZZY_TOKENS)

    score += exact_fourier_hits * 22.0
    score += min(fuzzy_fourier_hits, 4) * 8.0

    score += count_any_column(cols, IMAGE_LINK_COLUMNS) * 4.0
    score += count_any_column(cols, CLASS_COLUMNS) * 6.0
    score += count_any_column(cols, DEFECT_ID_COLUMNS) * 3.0
    score += count_any_column(cols, GEOMETRY_COLUMNS) * 2.2
    score += count_any_column(cols, SCORE_COLUMNS) * 2.5

    # BLOB columns often indicate coefficient storage or mask/geometry storage.
    blob_like = 0
    for col, typ in types.items():
        if "blob" in typ:
            blob_like += 1
        elif any(tok in col for tok in FOURIER_FUZZY_TOKENS):
            blob_like += 1

    score += min(blob_like, 4) * 5.0

    if has_any_column(cols, IMAGE_PATH_COLUMNS):
        # Defect rows often carry image path too. Slight positive, not too large.
        score += 2.0

    # Avoid obvious label table.
    label_score = score_label_table(schema, table)
    if label_score >= 40 and not has_fourier_column_by_schema(schema, table):
        score -= 24.0

    rows = table_row_count(schema, table)
    if rows > 0:
        score += min(5.0, math.log10(rows + 1.0) * 1.5)

    return score


def is_plausible_image_table(schema: Dict[str, Any], table: str) -> bool:
    """
    Plausibility gate for image table auto-detection.
    """
    name = table.lower()
    cols = schema_column_names(schema, table)

    exact = choose_table([table], IMAGE_TABLE_CANDIDATES) is not None
    name_hit = any(tok in name for tok in IMAGE_TABLE_NAME_TOKENS)
    path_hit = has_any_column(cols, IMAGE_PATH_COLUMNS)
    size_hit = count_any_column(cols, IMAGE_SIZE_COLUMNS) >= 1

    if exact:
        return True

    return (name_hit and (path_hit or size_hit)) or path_hit


def is_plausible_label_table(schema: Dict[str, Any], table: str) -> bool:
    """
    Plausibility gate for label table auto-detection.
    """
    name = table.lower()
    cols = schema_column_names(schema, table)

    exact = choose_table([table], LABEL_TABLE_CANDIDATES) is not None
    name_hit = any(tok in name for tok in LABEL_TABLE_NAME_TOKENS)

    id_hit = has_any_column(cols, ["class_id", "id", "label_id", "category_id", "cls_id"])
    name_col_hit = has_any_column(cols, ["class_name", "name", "label", "category"])

    if exact:
        return True

    return name_hit and id_hit and name_col_hit


def is_plausible_defect_table(schema: Dict[str, Any], table: str) -> bool:
    """
    Plausibility gate for defect table auto-detection.
    """
    name = table.lower()
    cols = schema_column_names(schema, table)

    exact = choose_table([table], DEFECT_TABLE_CANDIDATES) is not None
    name_hit = any(tok in name for tok in DEFECT_TABLE_NAME_TOKENS)

    has_coeff = has_fourier_column_by_schema(schema, table)
    has_image = has_any_column(cols, IMAGE_LINK_COLUMNS)
    has_class = has_any_column(cols, CLASS_COLUMNS)
    has_geometry = has_any_column(cols, GEOMETRY_COLUMNS)
    has_score = has_any_column(cols, SCORE_COLUMNS)

    if exact:
        return True

    if has_coeff and (has_image or has_class or has_geometry or has_score):
        return True

    if name_hit and has_image and (has_class or has_geometry or has_score):
        return True

    if has_image and has_class and (has_geometry or has_score):
        return True

    return False


def choose_best_table(
    schema: Dict[str, Any],
    candidates: Sequence[str],
    role: str,
) -> Optional[str]:
    """
    Choose table by exact candidate first, then schema scoring.

    role:
        "image_table"
        "defect_table"
        "label_table"
    """
    tables = list(schema.get("tables", []))

    # Exact candidate match always wins.
    exact = choose_table(tables, candidates)
    if exact:
        return exact

    scored: List[Dict[str, Any]] = []

    for table in tables:
        if role == "image_table":
            if not is_plausible_image_table(schema, table):
                continue
            score = score_image_table(schema, table)
        elif role == "defect_table":
            if not is_plausible_defect_table(schema, table):
                continue
            score = score_defect_table(schema, table)
        elif role == "label_table":
            if not is_plausible_label_table(schema, table):
                continue
            score = score_label_table(schema, table)
        else:
            score = 0.0

        scored.append(
            {
                "table": table,
                "score": round(float(score), 3),
                "rows": table_row_count(schema, table),
                "columns": schema_column_names(schema, table),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)

    schema.setdefault("table_scores", {})
    schema["table_scores"][role] = scored[:10]

    if not scored:
        return None

    best = scored[0]

    # Conservative thresholds.
    if role == "defect_table":
        return best["table"] if best["score"] >= 12.0 else None

    if role == "image_table":
        return best["table"] if best["score"] >= 8.0 else None

    if role == "label_table":
        return best["table"] if best["score"] >= 16.0 else None

    return None


def inspect_database(db_path: Path) -> Dict[str, Any]:
    """
    Inspect SQLite tables, columns, and row counts.
    """
    schema: Dict[str, Any] = {
        "db_path": str(db_path),
        "tables": [],
        "columns": {},
        "row_counts": {},
        "detected_tables": {
            "image_table": None,
            "defect_table": None,
            "label_table": None,
        },
        "table_scores": {},
    }

    with sqlite_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()

        tables = [str(r["name"]) for r in rows]
        schema["tables"] = tables

        for table in tables:
            pragma_rows = conn.execute(
                f"PRAGMA table_info({quote_identifier(table)})"
            ).fetchall()

            columns = [
                {
                    "cid": int(r["cid"]),
                    "name": str(r["name"]),
                    "type": str(r["type"] or ""),
                    "notnull": int(r["notnull"]),
                    "default_value": r["dflt_value"],
                    "pk": int(r["pk"]),
                }
                for r in pragma_rows
            ]
            schema["columns"][table] = columns

            try:
                count = conn.execute(
                    f"SELECT COUNT(*) AS n FROM {quote_identifier(table)}"
                ).fetchone()["n"]
                schema["row_counts"][table] = int(count)
            except Exception:
                schema["row_counts"][table] = None

    schema["detected_tables"]["image_table"] = choose_best_table(
        schema,
        IMAGE_TABLE_CANDIDATES,
        "image_table",
    )
    schema["detected_tables"]["defect_table"] = choose_best_table(
        schema,
        DEFECT_TABLE_CANDIDATES,
        "defect_table",
    )
    schema["detected_tables"]["label_table"] = choose_best_table(
        schema,
        LABEL_TABLE_CANDIDATES,
        "label_table",
    )

    return schema


# ---------------------------------------------------------------------
# Generic data helpers
# ---------------------------------------------------------------------
def normalize_path(path_like: Any, base_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Normalize a user path.

    If the path is relative and base_dir is provided, it will be resolved
    relative to base_dir.
    """
    if path_like is None:
        return None

    text = str(path_like).strip().strip('"').strip("'")
    if not text:
        return None

    p = Path(text).expanduser()

    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p

    try:
        return p.resolve()
    except Exception:
        return p


def is_nullish(value: Any) -> bool:
    """
    True if value should be treated as missing.
    """
    if value is None:
        return True

    if isinstance(value, str) and value.strip() == "":
        return True

    return False


def case_key_map(row: Dict[str, Any]) -> Dict[str, str]:
    """
    Map lowercase column name to real column name.
    """
    return {str(k).lower(): str(k) for k in row.keys()}


def get_first(row: Dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    """
    Get the first non-empty value among possible column names, case-insitively.
    """
    if not row:
        return default

    cmap = case_key_map(row)

    for name in names:
        real = cmap.get(str(name).lower())
        if real is not None:
            value = row.get(real)
            if not is_nullish(value):
                return value

    return default


def find_existing_key(row: Dict[str, Any], names: Sequence[str]) -> Tuple[Optional[str], Any]:
    """
    Return the real key and value for the first existing column.
    """
    if not row:
        return None, None

    cmap = case_key_map(row)

    for name in names:
        real = cmap.get(str(name).lower())
        if real is not None:
            return real, row.get(real)

    return None, None


def find_fourier_value(row: Dict[str, Any]) -> Tuple[Optional[str], Any]:
    """
    Find Fourier coefficient field in a row.

    First tries exact column names, then fuzzy names containing:
    fourier / coeff / descriptor / fsd / efd / harmonic / contour / shape.
    """
    key, value = find_existing_key(row, FOURIER_COLUMNS)
    if key is not None:
        return key, value

    for k, v in row.items():
        lk = str(k).lower()
        if any(tok in lk for tok in FOURIER_FUZZY_TOKENS):
            if isinstance(v, (bytes, memoryview, str, list, tuple, np.ndarray)) or v is not None:
                return str(k), v

    return None, None


def to_text(value: Any, default: Optional[str] = None) -> Optional[str]:
    """
    Convert a scalar value to text.
    """
    if is_nullish(value):
        return default

    try:
        return str(value)
    except Exception:
        return default


def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Convert a scalar value to float.
    """
    if is_nullish(value):
        return default

    try:
        v = float(value)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """
    Convert a scalar value to int.
    """
    if is_nullish(value):
        return default

    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def json_safe(value: Any) -> Any:
    """
    Convert Python/numpy/bytes values into JSON-safe values.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, bytes):
        return f"<BLOB {len(value)} bytes>"

    if isinstance(value, memoryview):
        b = bytes(value)
        return f"<BLOB {len(b)} bytes>"

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    try:
        return str(value)
    except Exception:
        return None


def sanitize_row(
    row: Dict[str, Any],
    *,
    skip_blob_columns: bool = True,
    skip_fourier_columns: bool = True,
) -> Dict[str, Any]:
    """
    Convert a raw SQLite row into a JSON-friendly dictionary.
    """
    out: Dict[str, Any] = {}

    fourier_lower = {c.lower() for c in FOURIER_COLUMNS}

    for k, v in row.items():
        key = str(k)
        key_lower = key.lower()

        if skip_fourier_columns:
            if key_lower in fourier_lower:
                continue
            if any(tok in key_lower for tok in FOURIER_FUZZY_TOKENS):
                continue

        if skip_blob_columns and isinstance(v, (bytes, memoryview)):
            continue

        out[key] = json_safe(v)

    return out


def stable_hash(text: str, length: int = 16) -> str:
    """
    Stable short SHA1 hash.
    """
    h = hashlib.sha1(str(text).encode("utf-8", errors="ignore")).hexdigest()
    return h[:length]


def make_image_key(
    *,
    image_pk: Any = None,
    image_uid: Any = None,
    image_id: Any = None,
    source_path: Any = None,
    index: int = 0,
) -> str:
    """
    Create a stable image key for frontend/API usage.
    """
    if not is_nullish(image_uid):
        return f"uid::{to_text(image_uid)}"

    if not is_nullish(image_pk):
        return f"pk::{to_text(image_pk)}"

    if not is_nullish(image_id):
        return f"id::{to_text(image_id)}"

    if not is_nullish(source_path):
        return f"path::{stable_hash(to_text(source_path) or '')}"

    return f"idx::{int(index)}"


# ---------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------
def load_label_map(conn: sqlite3.Connection, label_table: Optional[str]) -> Dict[str, str]:
    """
    Load label map from SQLite.

    Supports several possible column names:
    - class_id / id / label_id / category_id
    - class_name / name / label / category
    """
    label_map: Dict[str, str] = {}

    if not label_table:
        return label_map

    rows = read_all_rows(conn, label_table)

    for row in rows:
        class_id = get_first(row, ["class_id", "id", "label_id", "cls_id", "category_id"])
        class_name = get_first(row, ["class_name", "name", "label", "category"])

        if class_id is not None and class_name is not None:
            label_map[str(class_id)] = str(class_name)

    return label_map


# ---------------------------------------------------------------------
# Image file index and path resolution
# ---------------------------------------------------------------------
def build_image_file_index(image_dir: Optional[Path]) -> Dict[str, Any]:
    """
    Build a lightweight index of image files in image_dir.

    Important behavior for the viewer
    ---------------------------------
    If image_dir is selected by the user, the image browser should be driven
    by this folder, not by the database image table.

    Therefore this index stores:
    - files: stable ordered list of all image files in the selected folder
    - by_name: filename -> path
    - by_stem: stem -> path

    The database is still used for defect records and Fourier descriptors,
    but the visible image list is restricted to this selected folder.
    """
    index: Dict[str, Any] = {
        "enabled": False,
        "root": str(image_dir) if image_dir else None,
        "count": 0,
        "files": [],
        "by_name": {},
        "by_stem": {},
    }

    if image_dir is None or not image_dir.exists() or not image_dir.is_dir():
        return index

    index["enabled"] = True

    try:
        paths = sorted(
            image_dir.rglob("*"),
            key=lambda p: str(p).lower(),
        )

        for p in paths:
            if not p.is_file():
                continue

            suffix = p.suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue

            try:
                resolved = p.resolve()
            except Exception:
                resolved = p

            try:
                relative_path = str(resolved.relative_to(image_dir.resolve()))
            except Exception:
                relative_path = p.name

            name_key = p.name.lower()
            stem_key = p.stem.lower()

            file_item = {
                "path": str(resolved),
                "name": p.name,
                "stem": p.stem,
                "suffix": p.suffix,
                "relative_path": relative_path,
            }

            index["files"].append(file_item)

            # Keep the first match to make behavior stable.
            index["by_name"].setdefault(name_key, str(resolved))
            index["by_stem"].setdefault(stem_key, str(resolved))
            index["count"] += 1

    except Exception:
        # If scanning fails, fall back to direct path checks only.
        index["enabled"] = False
        index["files"] = []
        index["count"] = 0
        index["by_name"] = {}
        index["by_stem"] = {}

    return index


def make_image_from_folder_file(
    file_path: Path,
    image_dir: Path,
    index: int,
) -> Dict[str, Any]:
    """
    Create a normalized image entry directly from a file in the selected folder.

    This is the key change for folder-restricted display:
    - The visible image list comes from image_dir.
    - Every file in image_dir becomes one image-card in the frontend.
    - Defect records are attached later if database rows can be matched.
    """
    try:
        resolved = file_path.resolve()
    except Exception:
        resolved = file_path

    try:
        relative_path = str(resolved.relative_to(image_dir.resolve()))
    except Exception:
        relative_path = resolved.name

    image_key = f"folder::{stable_hash(str(resolved), length=20)}"

    return {
        "image_key": image_key,
        "image_pk": None,
        "image_uid": None,
        "image_id": resolved.stem,
        "source_path": relative_path,
        "img_w": None,
        "img_h": None,
        "det_count": 0,
        "classes": [],
        "max_score": None,
        "mean_score": None,
        "image_exists": True,
        "resolved_path": str(resolved),
        "meta": {
            "folder_source": True,
            "folder_index": int(index),
            "filename": resolved.name,
            "relative_path": relative_path,
            "absolute_path": str(resolved),
        },
    }


def build_folder_image_catalog(
    image_dir: Path,
    file_index: Dict[str, Any],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Dict[str, Any]]],
]:
    """
    Build the visible image catalog from the user-selected image folder.

    Unlike build_image_catalog(raw_images), this function ignores the database
    image table for display purposes. The selected folder is now the source of
    truth for the frontend image browser.
    """
    images: List[Dict[str, Any]] = []
    images_by_key: Dict[str, Dict[str, Any]] = {}

    maps: Dict[str, Dict[str, Dict[str, Any]]] = {
        "by_pk": {},
        "by_uid": {},
        "by_id": {},
        "by_filename": {},
        "by_stem": {},
    }

    files = file_index.get("files", [])

    for i, item in enumerate(files):
        path_text = item.get("path")
        if not path_text:
            continue

        p = Path(str(path_text))
        if not p.exists() or not p.is_file():
            continue

        image = make_image_from_folder_file(p, image_dir, i)
        register_image(image, images, images_by_key, maps)

    return images, images_by_key, maps


def apply_database_image_aliases_to_folder_maps(
    raw_images: List[Dict[str, Any]],
    maps: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """
    Add database image-table aliases to folder images.

    Why this is needed
    ------------------
    In many SQLite archives, defect rows do not store the image filename
    directly. They only store image_pk / image_id / image_uid.

    But after switching to folder-driven image display, visible image entries
    are created from files, so they initially have no database image_pk.

    This function bridges them:

        DB image row path/name/stem -> selected folder image
        DB image_pk/image_uid/image_id -> selected folder image

    Result:
    - Frontend still only shows selected-folder images.
    - Defect records using image_pk can still attach to those folder images.
    """
    if not raw_images:
        return

    for i, row in enumerate(raw_images):
        db_image = normalize_image_row(row, i)

        target: Optional[Dict[str, Any]] = None

        source_path = to_text(db_image.get("source_path"))
        image_id = to_text(db_image.get("image_id"))

        if source_path:
            p = Path(source_path)
            target = (
                maps["by_filename"].get(p.name.lower())
                or maps["by_stem"].get(p.stem.lower())
            )

        if target is None and image_id:
            p = Path(image_id)
            target = (
                maps["by_filename"].get(p.name.lower())
                or maps["by_stem"].get(p.stem.lower())
                or maps["by_stem"].get(str(image_id).lower())
            )

        if target is None:
            continue

        image_pk = db_image.get("image_pk")
        image_uid = db_image.get("image_uid")
        db_image_id = db_image.get("image_id")
        db_source_path = db_image.get("source_path")

        if not is_nullish(image_pk):
            maps["by_pk"].setdefault(str(image_pk), target)

        if not is_nullish(image_uid):
            maps["by_uid"].setdefault(str(image_uid), target)

        if not is_nullish(db_image_id):
            maps["by_id"].setdefault(str(db_image_id), target)

            p = Path(str(db_image_id))
            maps["by_filename"].setdefault(p.name.lower(), target)
            maps["by_stem"].setdefault(p.stem.lower(), target)

        if not is_nullish(db_source_path):
            p = Path(str(db_source_path))
            maps["by_filename"].setdefault(p.name.lower(), target)
            maps["by_stem"].setdefault(p.stem.lower(), target)


def resolve_image_path(
    image: Dict[str, Any],
    image_dir: Optional[Path],
    file_index: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Resolve the actual local image file path.

    Search priority:
    1. Absolute source_path if it exists.
    2. image_dir / relative source_path.
    3. image_dir / basename(source_path).
    4. Indexed filename match.
    5. Indexed stem/image_id match.
    6. image_dir / image_id + known extension.
    """
    if image is None:
        return None

    resolved_path = to_text(image.get("resolved_path"))
    if resolved_path:
        p = Path(resolved_path)
        try:
            if p.exists() and p.is_file():
                return p.resolve()
        except Exception:
            pass

    source_path = to_text(image.get("source_path"))
    image_id = to_text(image.get("image_id"))
    candidates: List[Path] = []

    # 1. Source path direct.
    if source_path:
        p = Path(source_path).expanduser()

        if p.is_absolute():
            candidates.append(p)
        else:
            if image_dir is not None:
                candidates.append(image_dir / p)
            candidates.append(SQLITE_DIR / p)
            candidates.append(PROJECT_DIR / p)

        # 2. image_dir / basename(source_path)
        if image_dir is not None:
            candidates.append(image_dir / p.name)

    # 3. image_id + extension.
    if image_dir is not None and image_id:
        image_id_path = Path(str(image_id))
        candidates.append(image_dir / image_id_path.name)

        for ext in IMAGE_EXTENSIONS:
            candidates.append(image_dir / f"{image_id}{ext}")
            candidates.append(image_dir / f"{image_id_path.stem}{ext}")

    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return c.resolve()
        except Exception:
            pass

    # 4. Index lookup.
    if file_index and file_index.get("enabled"):
        by_name = file_index.get("by_name", {})
        by_stem = file_index.get("by_stem", {})

        if source_path:
            name = Path(source_path).name.lower()
            stem = Path(source_path).stem.lower()

            hit = by_name.get(name) or by_stem.get(stem)
            if hit:
                p = Path(hit)
                if p.exists():
                    return p.resolve()

        if image_id:
            p = Path(str(image_id))
            hit = by_name.get(p.name.lower()) or by_stem.get(p.stem.lower()) or by_stem.get(str(image_id).lower())
            if hit:
                pp = Path(hit)
                if pp.exists():
                    return pp.resolve()

    return None


# ---------------------------------------------------------------------
# Image normalization
# ---------------------------------------------------------------------
def normalize_image_row(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Normalize one image row from an image table.
    """
    image_pk = get_first(row, ["image_pk", "pk", "id", "image_id_pk"])
    image_uid = get_first(row, ["image_uid", "uid", "uuid", "image_uuid"])
    image_id = get_first(
        row,
        ["image_id", "img_id", "image_name", "img_name", "name", "file_id", "filename", "file_name"],
    )
    source_path = get_first(
        row,
        ["source_path", "image_path", "file_path", "path", "relative_path", "filename", "file_name", "image_file"],
    )

    if is_nullish(image_id) and not is_nullish(source_path):
        image_id = Path(str(source_path)).stem

    img_w = to_int(get_first(row, ["img_w", "image_width", "width", "w"]))
    img_h = to_int(get_first(row, ["img_h", "image_height", "height", "h"]))

    image_key = make_image_key(
        image_pk=image_pk,
        image_uid=image_uid,
        image_id=image_id,
        source_path=source_path,
        index=index,
    )

    return {
        "image_key": image_key,
        "image_pk": json_safe(image_pk),
        "image_uid": json_safe(image_uid),
        "image_id": to_text(image_id, default=f"image_{index:06d}"),
        "source_path": to_text(source_path),
        "img_w": img_w,
        "img_h": img_h,
        "det_count": 0,
        "classes": [],
        "max_score": None,
        "mean_score": None,
        "image_exists": False,
        "resolved_path": None,
        "meta": sanitize_row(row),
    }


def register_image(
    image: Dict[str, Any],
    images: List[Dict[str, Any]],
    images_by_key: Dict[str, Dict[str, Any]],
    maps: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Register one image into image list and lookup maps.
    """
    key = str(image.get("image_key"))

    if key in images_by_key:
        base = key
        i = 2

        while f"{base}#{i}" in images_by_key:
            i += 1

        key = f"{base}#{i}"
        image["image_key"] = key

    images.append(image)
    images_by_key[key] = image

    image_pk = image.get("image_pk")
    image_uid = image.get("image_uid")
    image_id = image.get("image_id")
    source_path = image.get("source_path")
    resolved_path = image.get("resolved_path")

    if not is_nullish(image_pk):
        maps["by_pk"].setdefault(str(image_pk), image)

    if not is_nullish(image_uid):
        maps["by_uid"].setdefault(str(image_uid), image)

    if not is_nullish(image_id):
        maps["by_id"].setdefault(str(image_id), image)
        p = Path(str(image_id))
        maps["by_filename"].setdefault(p.name.lower(), image)
        maps["by_stem"].setdefault(p.stem.lower(), image)

    if not is_nullish(source_path):
        p = Path(str(source_path))
        maps["by_filename"].setdefault(p.name.lower(), image)
        maps["by_stem"].setdefault(p.stem.lower(), image)

    if not is_nullish(resolved_path):
        p = Path(str(resolved_path))
        maps["by_filename"].setdefault(p.name.lower(), image)
        maps["by_stem"].setdefault(p.stem.lower(), image)

    return image


def build_image_catalog(raw_images: List[Dict[str, Any]]) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Dict[str, Any]]],
]:
    """
    Build normalized image catalog and lookup maps.
    """
    images: List[Dict[str, Any]] = []
    images_by_key: Dict[str, Dict[str, Any]] = {}

    maps: Dict[str, Dict[str, Dict[str, Any]]] = {
        "by_pk": {},
        "by_uid": {},
        "by_id": {},
        "by_filename": {},
        "by_stem": {},
    }

    for i, row in enumerate(raw_images):
        image = normalize_image_row(row, i)
        register_image(image, images, images_by_key, maps)

    return images, images_by_key, maps


def match_image_for_defect(
    row: Dict[str, Any],
    maps: Dict[str, Dict[str, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """
    Match a defect row to a normalized image entry.
    """
    image_pk = get_first(row, ["image_pk", "image_fk", "img_pk", "img_fk"])
    image_uid = get_first(row, ["image_uid", "img_uid", "image_uuid", "uid", "uuid"])
    image_id = get_first(
        row,
        ["image_id", "img_id", "image_name", "img_name", "name", "file_id", "filename", "file_name"],
    )
    source_path = get_first(
        row,
        ["source_path", "image_path", "file_path", "path", "relative_path", "filename", "file_name", "image_file"],
    )

    if not is_nullish(image_pk):
        hit = maps["by_pk"].get(str(image_pk))
        if hit is not None:
            return hit

    if not is_nullish(image_uid):
        hit = maps["by_uid"].get(str(image_uid))
        if hit is not None:
            return hit

    if not is_nullish(image_id):
        hit = maps["by_id"].get(str(image_id))
        if hit is not None:
            return hit

        p = Path(str(image_id))
        hit = maps["by_filename"].get(p.name.lower())
        if hit is not None:
            return hit

        hit = maps["by_stem"].get(p.stem.lower())
        if hit is not None:
            return hit

    if not is_nullish(source_path):
        p = Path(str(source_path))

        hit = maps["by_filename"].get(p.name.lower())
        if hit is not None:
            return hit

        hit = maps["by_stem"].get(p.stem.lower())
        if hit is not None:
            return hit

    return None


def create_image_from_defect(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Create a minimal image entry from a defect row when image table is missing
    or when this defect cannot be matched.
    """
    image_pk = get_first(row, ["image_pk", "image_fk", "img_pk", "img_fk"])
    image_uid = get_first(row, ["image_uid", "img_uid", "image_uuid", "uid", "uuid"])
    image_id = get_first(
        row,
        ["image_id", "img_id", "image_name", "img_name", "name", "file_id", "filename", "file_name"],
    )
    source_path = get_first(
        row,
        ["source_path", "image_path", "file_path", "path", "relative_path", "filename", "file_name", "image_file"],
    )

    if is_nullish(image_id) and not is_nullish(source_path):
        image_id = Path(str(source_path)).stem

    img_w = to_int(get_first(row, ["img_w", "image_width", "width", "w"]))
    img_h = to_int(get_first(row, ["img_h", "image_height", "height", "h"]))

    image_key = make_image_key(
        image_pk=image_pk,
        image_uid=image_uid,
        image_id=image_id,
        source_path=source_path,
        index=index,
    )

    return {
        "image_key": image_key,
        "image_pk": json_safe(image_pk),
        "image_uid": json_safe(image_uid),
        "image_id": to_text(image_id, default=f"image_{index:06d}"),
        "source_path": to_text(source_path),
        "img_w": img_w,
        "img_h": img_h,
        "det_count": 0,
        "classes": [],
        "max_score": None,
        "mean_score": None,
        "image_exists": False,
        "resolved_path": None,
        "meta": {},
    }


# ---------------------------------------------------------------------
# Fourier coefficient decoding
# ---------------------------------------------------------------------
def normalize_codec(codec: Any) -> Optional[str]:
    """
    Normalize coefficient codec name.
    """
    if codec is None:
        return None

    c = str(codec).strip().lower()

    if c in {"f16", "float16", "fp16", "half"}:
        return "f16"

    if c in {"f32", "float32", "fp32", "single"}:
        return "f32"

    if c in {"f64", "float64", "fp64", "double"}:
        return "f64"

    if c in {"json", "list", "array"}:
        return "json"

    return c or None


def infer_codec(value: Any, explicit_codec: Any = None, expected_count: Any = None) -> Optional[str]:
    """
    Infer codec when possible.

    FS-FSD usually stores coefficients as float16 BLOB, so f16 is used as
    the final binary fallback.
    """
    codec = normalize_codec(explicit_codec)

    if codec:
        return codec

    if value is None:
        return None

    if isinstance(value, str):
        return "json"

    if isinstance(value, (list, tuple, np.ndarray)):
        return "array"

    if isinstance(value, memoryview):
        value = bytes(value)

    if isinstance(value, bytes):
        head = value[:1]

        if head in {b"[", b"{"}:
            return "json"

        n = len(value)
        count = to_int(expected_count)

        if count and count > 0:
            if n == count * 2:
                return "f16"
            if n == count * 4:
                return "f32"
            if n == count * 8:
                return "f64"

        # Default based on current FS-FSD storage.
        if n % 2 == 0:
            return "f16"

        if n % 4 == 0:
            return "f32"

    return None


def decode_coefficients_value(
    value: Any,
    codec: Any = None,
    expected_count: Any = None,
) -> Optional[np.ndarray]:
    """
    Decode Fourier coefficients into a 1D float32 numpy array.
    """
    if value is None:
        return None

    inferred = infer_codec(value, codec, expected_count)

    try:
        if isinstance(value, memoryview):
            value = bytes(value)

        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.float32).reshape(-1)

        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=np.float32).reshape(-1)

        if isinstance(value, str):
            data = json.loads(value)
            return np.asarray(data, dtype=np.float32).reshape(-1)

        if isinstance(value, bytes):
            if inferred == "json":
                data = json.loads(value.decode("utf-8"))
                return np.asarray(data, dtype=np.float32).reshape(-1)

            if inferred == "f16":
                if len(value) % np.dtype(np.float16).itemsize != 0:
                    return None
                return np.frombuffer(value, dtype=np.float16).astype(np.float32).reshape(-1)

            if inferred == "f32":
                if len(value) % np.dtype(np.float32).itemsize != 0:
                    return None
                return np.frombuffer(value, dtype=np.float32).astype(np.float32).reshape(-1)

            if inferred == "f64":
                if len(value) % np.dtype(np.float64).itemsize != 0:
                    return None
                return np.frombuffer(value, dtype=np.float64).astype(np.float32).reshape(-1)

    except Exception:
        return None

    return None


def decode_coefficients_from_record(record: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Decode coefficients from a normalized internal defect record.
    """
    value = record.get("_coeff_data")
    codec = record.get("coeff_codec")
    expected_count = record.get("coeff_count")

    return decode_coefficients_value(value, codec, expected_count)


# ---------------------------------------------------------------------
# Polygon reconstruction
# ---------------------------------------------------------------------
def coerce_polygon(obj: Any) -> Optional[np.ndarray]:
    """
    Convert a polygon-like return object to an array of shape [N, 2].
    """
    if obj is None:
        return None

    if isinstance(obj, dict):
        for key in ["polygon", "points", "contour", "vertices", "coords"]:
            if key in obj:
                out = coerce_polygon(obj.get(key))
                if out is not None:
                    return out

    try:
        arr = np.asarray(obj, dtype=np.float32)

        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]

        if arr.ndim == 1 and arr.size >= 6 and arr.size % 2 == 0:
            return arr.reshape(-1, 2)
    except Exception:
        pass

    if isinstance(obj, (tuple, list)):
        for item in obj:
            out = coerce_polygon(item)
            if out is not None:
                return out

    return None


def call_project_geometry_reconstruction(
    coeffs: np.ndarray,
    img_w: Optional[int],
    img_h: Optional[int],
    num_points: int,
) -> Optional[np.ndarray]:
    """
    Try calling the existing fsd_geometry.reconstruct_polygon_from_coeffs.

    The exact function signature can differ slightly between project versions,
    so multiple call styles are attempted safely.
    """
    if _fsd_reconstruct_polygon is None:
        return None

    attempts = []

    attempts.append(lambda: _fsd_reconstruct_polygon(coeffs, img_w, img_h, num_points))
    attempts.append(lambda: _fsd_reconstruct_polygon(coeffs, img_w=img_w, img_h=img_h, num_points=num_points))
    attempts.append(lambda: _fsd_reconstruct_polygon(coeffs, image_width=img_w, image_height=img_h, num_points=num_points))
    attempts.append(lambda: _fsd_reconstruct_polygon(coeffs, width=img_w, height=img_h, num_points=num_points))
    attempts.append(lambda: _fsd_reconstruct_polygon(coeffs, num_points=num_points))
    attempts.append(lambda: _fsd_reconstruct_polygon(coeffs))

    for fn in attempts:
        try:
            out = fn()
            poly = coerce_polygon(out)

            if poly is not None and len(poly) >= 3:
                return poly
        except Exception:
            continue

    return None


def fallback_polygon_from_metrics(
    record: Dict[str, Any],
    num_points: int,
    img_w: Optional[int],
    img_h: Optional[int],
) -> np.ndarray:
    """
    Last-resort polygon reconstruction.

    This is only used if the project geometry function is unavailable or fails.
    It creates an ellipse-like polygon from centroid/area/orientation metadata.
    """
    cx = to_float(record.get("centroid_x_px"))
    cy = to_float(record.get("centroid_y_px"))

    if cx is None:
        cx = (img_w or 1000) / 2.0

    if cy is None:
        cy = (img_h or 1000) / 2.0

    area = to_float(record.get("area_px2"), 900.0) or 900.0
    elongation = to_float(record.get("elongation"), 2.0) or 2.0
    orientation = math.radians(to_float(record.get("orientation_deg"), 0.0) or 0.0)

    elongation = max(1.0, min(float(elongation), 20.0))

    # Ellipse area = pi * rx * ry, rx / ry = elongation.
    ry = math.sqrt(max(area, 1.0) / (math.pi * elongation))
    rx = ry * elongation

    n = max(16, int(num_points))
    pts = []

    for i in range(n):
        a = 2.0 * math.pi * i / n

        # Slight deformation so the fallback still looks like damage contour,
        # not a perfect CAD ellipse.
        wave = 1.0 + 0.07 * math.sin(3 * a) + 0.04 * math.sin(7 * a)

        ex = math.cos(a) * rx * wave
        ey = math.sin(a) * ry * wave

        x = cx + ex * math.cos(orientation) - ey * math.sin(orientation)
        y = cy + ex * math.sin(orientation) + ey * math.cos(orientation)

        pts.append([x, y])

    return np.asarray(pts, dtype=np.float32)


def fallback_fourier_polygon(
    coeffs: np.ndarray,
    record: Dict[str, Any],
    num_points: int,
    img_w: Optional[int],
    img_h: Optional[int],
) -> np.ndarray:
    """
    Generic Fourier fallback.

    It supports a common layout:
        [cx, cy, ax1, bx1, ay1, by1, ax2, bx2, ay2, by2, ...]

    If the coefficient layout does not match this assumption, it falls back
    to a metadata-based ellipse.
    """
    arr = np.asarray(coeffs, dtype=np.float32).reshape(-1)

    n = max(16, int(num_points))

    if arr.size >= 6 and (arr.size - 2) % 4 == 0:
        cx = float(arr[0])
        cy = float(arr[1])
        terms = arr[2:].reshape(-1, 4)
    elif arr.size >= 4 and arr.size % 4 == 0:
        cx = to_float(record.get("centroid_x_px"))
        cy = to_float(record.get("centroid_y_px"))

        if cx is None:
            cx = (img_w or 1000) / 2.0

        if cy is None:
            cy = (img_h or 1000) / 2.0

        terms = arr.reshape(-1, 4)
    else:
        return fallback_polygon_from_metrics(record, n, img_w, img_h)

    t = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float32)
    x = np.full_like(t, cx, dtype=np.float32)
    y = np.full_like(t, cy, dtype=np.float32)

    for k, row in enumerate(terms, start=1):
        ax, bx, ay, by = [float(v) for v in row]

        x += ax * np.cos(k * t) + bx * np.sin(k * t)
        y += ay * np.cos(k * t) + by * np.sin(k * t)

    # If coefficients appear normalized, scale to pixel domain.
    if img_w and img_h:
        finite_x = x[np.isfinite(x)]
        finite_y = y[np.isfinite(y)]

        if finite_x.size and finite_y.size:
            max_abs = max(
                float(np.max(np.abs(finite_x))),
                float(np.max(np.abs(finite_y))),
            )

            if max_abs <= 3.0:
                x = x * float(img_w)
                y = y * float(img_h)

    return np.stack([x, y], axis=1).astype(np.float32)


def sanitize_polygon(poly: np.ndarray) -> List[List[float]]:
    """
    Convert polygon array to clean JSON-ready list.
    """
    arr = np.asarray(poly, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] < 2:
        return []

    out: List[List[float]] = []

    for x, y in arr[:, :2]:
        xf = float(x)
        yf = float(y)

        if math.isfinite(xf) and math.isfinite(yf):
            out.append([round(xf, 3), round(yf, 3)])

    return out


def reconstruct_polygon_for_record(
    record: Dict[str, Any],
    num_points: Optional[int] = None,
) -> List[List[float]]:
    """
    Reconstruct one defect polygon.
    """
    n = to_int(
        num_points,
        default=to_int(record.get("poly_num_points"), default=128),
    ) or 128

    n = max(16, min(int(n), 1024))

    img_w = to_int(record.get("img_w"))
    img_h = to_int(record.get("img_h"))

    coeffs = decode_coefficients_from_record(record)

    if coeffs is not None and coeffs.size >= 4:
        poly = call_project_geometry_reconstruction(coeffs, img_w, img_h, n)

        if poly is None:
            poly = fallback_fourier_polygon(coeffs, record, n, img_w, img_h)
    else:
        poly = fallback_polygon_from_metrics(record, n, img_w, img_h)

    return sanitize_polygon(poly)


# ---------------------------------------------------------------------
# Defect normalization
# ---------------------------------------------------------------------
def normalize_defect_row(
    row: Dict[str, Any],
    index: int,
    image: Dict[str, Any],
    label_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    Normalize one defect row.
    """
    coeff_key, coeff_value = find_fourier_value(row)

    defect_id = get_first(
        row,
        [
            "defect_id",
            "defect_pk",
            "damage_id",
            "annotation_id",
            "instance_id",
            "object_id",
            "detection_id",
            "id",
            "pk",
        ],
        default=index,
    )

    class_id = get_first(row, ["class_id", "cls_id", "category_id", "label_id"])
    class_name = get_first(
        row,
        ["class_name", "category", "label", "name", "damage_type", "defect_type"],
    )

    if is_nullish(class_name) and class_id is not None:
        class_name = label_map.get(str(class_id))

    if is_nullish(class_name):
        if class_id is not None:
            class_name = f"class_{class_id}"
        else:
            class_name = "unknown"

    score = to_float(get_first(row, ["score", "confidence", "conf", "probability", "prob"]))

    img_w = to_int(
        get_first(row, ["img_w", "image_width", "width", "w"]),
        default=to_int(image.get("img_w")),
    )
    img_h = to_int(
        get_first(row, ["img_h", "image_height", "height", "h"]),
        default=to_int(image.get("img_h")),
    )

    explicit_codec = get_first(row, ["coeff_codec", "codec", "fourier_codec", "coeff_dtype", "dtype"])
    explicit_count = get_first(row, ["coeff_count", "num_coeffs", "coefficient_count", "n_coeffs"])

    codec = infer_codec(coeff_value, explicit_codec, explicit_count)
    decoded = decode_coefficients_value(coeff_value, codec, explicit_count)
    coeff_count = int(decoded.size) if decoded is not None else to_int(explicit_count)

    coeff_bytes = None

    if isinstance(coeff_value, memoryview):
        coeff_bytes = len(bytes(coeff_value))
    elif isinstance(coeff_value, bytes):
        coeff_bytes = len(coeff_value)
    elif isinstance(coeff_value, str):
        coeff_bytes = len(coeff_value.encode("utf-8"))

    rec = sanitize_row(row, skip_blob_columns=True, skip_fourier_columns=True)

    # Canonical fields used by the WebGL viewer.
    rec.update(
        {
            "defect_id": to_text(defect_id, default=str(index)),
            "image_key": image.get("image_key"),
            "image_id": image.get("image_id"),
            "image_uid": image.get("image_uid"),
            "source_path": image.get("source_path"),
            "img_w": img_w,
            "img_h": img_h,
            "class_id": json_safe(class_id),
            "class_name": to_text(class_name, default="unknown"),
            "score": score,
            "centroid_x_px": to_float(
                get_first(row, ["centroid_x_px", "cx", "center_x", "x_center"])
            ),
            "centroid_y_px": to_float(
                get_first(row, ["centroid_y_px", "cy", "center_y", "y_center"])
            ),
            "area_px2": to_float(get_first(row, ["area_px2", "area", "mask_area"])),
            "perimeter_px": to_float(get_first(row, ["perimeter_px", "perimeter"])),
            "orientation_deg": to_float(
                get_first(row, ["orientation_deg", "angle_deg", "orientation"])
            ),
            "elongation": to_float(get_first(row, ["elongation", "aspect_ratio"])),
            "fourier_order": to_int(get_first(row, ["fourier_order", "order", "harmonic_order", "n_harmonics"])),
            "coeff_codec": codec,
            "coeff_count": coeff_count,
            "coeff_bytes": coeff_bytes,
            "coeff_column": coeff_key,
            "poly_num_points": to_int(
                get_first(row, ["poly_num_points", "polygon_points", "num_polygon_points"]),
                default=128,
            ),
            "model_name": to_text(get_first(row, ["model_name", "model"])),
            "model_version": to_text(get_first(row, ["model_version", "version"])),
            "inspection_id": to_text(get_first(row, ["inspection_id", "inspection"])),
            "bridge_id": to_text(get_first(row, ["bridge_id", "bridge"])),
            "component_id": to_text(get_first(row, ["component_id", "component"])),
        }
    )

    # Internal fields. These are not sent directly to the frontend.
    rec["_raw_index"] = int(index)
    rec["_coeff_data"] = coeff_value
    rec["_coeff_column"] = coeff_key

    return rec


def update_image_statistics(
    images: List[Dict[str, Any]],
    defects_by_image_key: Dict[str, List[Dict[str, Any]]],
    image_dir: Optional[Path],
    file_index: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Update det_count/classes/max_score/resolved_path for all images.
    """
    resolved: Dict[str, str] = {}

    for img in images:
        key = str(img.get("image_key"))
        records = defects_by_image_key.get(key, [])

        scores = [
            to_float(r.get("score"))
            for r in records
            if to_float(r.get("score")) is not None
        ]

        classes = sorted(
            {
                str(r.get("class_name"))
                for r in records
                if not is_nullish(r.get("class_name"))
            }
        )

        img["det_count"] = len(records)
        img["classes"] = classes
        img["max_score"] = round(max(scores), 6) if scores else None
        img["mean_score"] = round(sum(scores) / len(scores), 6) if scores else None

        path = resolve_image_path(img, image_dir, file_index)
        if path is not None:
            img["image_exists"] = True
            img["resolved_path"] = str(path)
            resolved[key] = str(path)
        else:
            img["image_exists"] = False
            img["resolved_path"] = None

    return resolved


# ---------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------
def build_dataset(
    raw_images: List[Dict[str, Any]],
    raw_defects: List[Dict[str, Any]],
    label_map: Dict[str, str],
    image_dir: Optional[Path],
    file_index: Optional[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, List[Dict[str, Any]]],
    Dict[str, Dict[str, Any]],
    Dict[str, str],
]:
    """
    Build normalized image and defect datasets.

    Display policy
    --------------
    If the user selected an image_dir and that folder contains image files,
    the frontend image list is restricted to that folder.

    This means:

    - Visible images = files inside selected image_dir.
    - Database image rows outside image_dir are not displayed.
    - Database defects are attached only when they match a selected-folder image.
    - Folder images without matching defects are still displayed with det_count=0.

    If image_dir is not selected or contains no images, the previous database-
    driven behavior is preserved.
    """
    folder_mode = (
        image_dir is not None
        and file_index is not None
        and bool(file_index.get("enabled"))
        and int(file_index.get("count", 0) or 0) > 0
    )

    if folder_mode:
        # New behavior:
        # The selected folder is the source of truth for visible images.
        images, images_by_key, maps = build_folder_image_catalog(
            image_dir=image_dir,
            file_index=file_index,
        )

        # Add DB aliases so defects using image_pk/image_uid/image_id can still
        # attach to the corresponding folder image.
        apply_database_image_aliases_to_folder_maps(raw_images, maps)

    else:
        # Original fallback behavior:
        # Use database image table as the image catalog.
        images, images_by_key, maps = build_image_catalog(raw_images)

    defects_by_image_key: Dict[str, List[Dict[str, Any]]] = {
        str(img.get("image_key")): []
        for img in images
    }
    defects_by_id: Dict[str, Dict[str, Any]] = {}

    for i, row in enumerate(raw_defects):
        image = match_image_for_defect(row, maps)

        if image is None:
            if folder_mode:
                # Folder-restricted mode:
                # Do NOT create a new visible image from a defect row.
                # This prevents database images outside the selected folder
                # from appearing in the frontend.
                continue

            # Original behavior:
            # If no image row is available or matching fails, create one from
            # defect metadata.
            candidate = create_image_from_defect(row, i)
            candidate_key = str(candidate.get("image_key"))

            existing = images_by_key.get(candidate_key)
            if existing is not None:
                image = existing
            else:
                image = register_image(candidate, images, images_by_key, maps)
                defects_by_image_key.setdefault(str(image.get("image_key")), [])

        image_key = str(image.get("image_key"))
        rec = normalize_defect_row(row, i, image, label_map)

        # Make defect_id unique if necessary.
        defect_id = str(rec.get("defect_id"))

        if defect_id in defects_by_id:
            defect_id = f"{defect_id}#{i}"
            rec["defect_id"] = defect_id

        defects_by_image_key.setdefault(image_key, []).append(rec)
        defects_by_id[defect_id] = rec

    if folder_mode:
        # Folder images are already resolved by construction.
        resolved = {
            str(img.get("image_key")): str(img.get("resolved_path"))
            for img in images
            if img.get("resolved_path")
        }

        # Still update defect statistics for selected-folder images.
        for img in images:
            key = str(img.get("image_key"))
            records = defects_by_image_key.get(key, [])

            scores = [
                to_float(r.get("score"))
                for r in records
                if to_float(r.get("score")) is not None
            ]

            classes = sorted(
                {
                    str(r.get("class_name"))
                    for r in records
                    if not is_nullish(r.get("class_name"))
                }
            )

            img["det_count"] = len(records)
            img["classes"] = classes
            img["max_score"] = round(max(scores), 6) if scores else None
            img["mean_score"] = round(sum(scores) / len(scores), 6) if scores else None
            img["image_exists"] = True

    else:
        resolved = update_image_statistics(
            images,
            defects_by_image_key,
            image_dir,
            file_index,
        )

    return images, defects_by_image_key, defects_by_id, resolved


def compute_summary(
    *,
    db_path: Path,
    image_dir: Optional[Path],
    schema: Dict[str, Any],
    images: List[Dict[str, Any]],
    defects_by_id: Dict[str, Dict[str, Any]],
    file_index: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute database session summary.
    """
    defect_records = list(defects_by_id.values())

    classes = sorted(
        {
            str(r.get("class_name"))
            for r in defect_records
            if not is_nullish(r.get("class_name"))
        }
    )

    codecs = sorted(
        {
            str(r.get("coeff_codec"))
            for r in defect_records
            if not is_nullish(r.get("coeff_codec"))
        }
    )

    total_coeff_bytes = 0

    for r in defect_records:
        b = to_int(r.get("coeff_bytes"))
        if b is not None:
            total_coeff_bytes += b

    db_file_size = db_path.stat().st_size if db_path.exists() else None

    image_found_count = sum(1 for img in images if img.get("image_exists"))
    missing_image_count = max(0, len(images) - image_found_count)

    folder_source_count = sum(
        1
        for img in images
        if isinstance(img.get("meta"), dict) and img.get("meta", {}).get("folder_source")
    )

    display_mode = (
        "selected_image_folder_only"
        if image_dir is not None and folder_source_count > 0
        else "database_image_catalog"
    )

    detected_tables = schema.get("detected_tables", {})

    return {
        "db_path": str(db_path),
        "image_dir": str(image_dir) if image_dir else None,
        "display_mode": display_mode,
        "folder_source_image_count": folder_source_count,
        "db_file_size": db_file_size,
        "db_file_size_mb": round(db_file_size / (1024 * 1024), 4) if db_file_size else None,
        "tables": schema.get("tables", []),
        "row_counts": schema.get("row_counts", {}),
        "detected_tables": detected_tables,
        "table_scores": schema.get("table_scores", {}),
        "image_table": detected_tables.get("image_table"),
        "defect_table": detected_tables.get("defect_table"),
        "label_table": detected_tables.get("label_table"),
        "image_count": len(images),
        "defect_count": len(defects_by_id),
        "class_count": len(classes),
        "classes": classes,
        "coefficient_codecs": codecs,
        "total_coeff_bytes": total_coeff_bytes,
        "total_coeff_kb": round(total_coeff_bytes / 1024, 3),
        "image_found_count": image_found_count,
        "missing_image_count": missing_image_count,
        "indexed_image_files": int(file_index.get("count", 0)) if file_index else 0,
        "geometry_backend": "fsd_geometry.reconstruct_polygon_from_coeffs"
        if _fsd_reconstruct_polygon is not None
        else "internal fallback",
    }


def format_schema_detection_error(schema: Dict[str, Any]) -> str:
    """
    Build a helpful error message when defect table cannot be detected.
    """
    tables = schema.get("tables", [])
    row_counts = schema.get("row_counts", {})
    columns = schema.get("columns", {})
    scores = schema.get("table_scores", {}).get("defect_table", [])

    table_lines = []

    for table in tables:
        col_names = [
            str(c.get("name", ""))
            for c in columns.get(table, [])
        ]
        table_lines.append(
            f"- {table} rows={row_counts.get(table)} columns={col_names[:18]}"
        )

    score_lines = []
    for item in scores:
        score_lines.append(
            f"- {item.get('table')} score={item.get('score')} rows={item.get('rows')}"
        )

    message = (
        "No defect/damage table could be detected automatically.\n\n"
        "The selected SQLite database does not expose one of the common table names, "
        "and its columns did not match the FS-FSD defect schema strongly enough.\n\n"
        "Detected tables:\n"
        + "\n".join(table_lines[:30])
    )

    if score_lines:
        message += "\n\nTop defect-table candidates:\n" + "\n".join(score_lines[:10])

    message += (
        "\n\nPlease send the table/column list if this still fails, and I will map "
        "your exact FS-FSD archive schema."
    )

    return message


# ---------------------------------------------------------------------
# Public service API
# ---------------------------------------------------------------------
def open_archive(db_path: Any, image_dir: Any = None) -> Dict[str, Any]:
    """
    Open an FS-FSD SQLite archive and load it into VIEWER_STATE.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    image_dir:
        Optional path to source image directory.

    Returns
    -------
    dict
        Summary dictionary.
    """
    db = normalize_path(db_path, base_dir=SQLITE_DIR)

    if db is None:
        raise FSDServiceError("Database path is empty.")

    if not db.exists() or not db.is_file():
        raise FSDServiceError(f"Database file does not exist: {db}")

    img_dir = normalize_path(image_dir, base_dir=SQLITE_DIR) if image_dir else None

    if img_dir is not None and not img_dir.exists():
        # Do not fail hard. The database can still be inspected without images.
        img_dir_exists = False
    else:
        img_dir_exists = True

    try:
        schema = inspect_database(db)

        defect_table = schema.get("detected_tables", {}).get("defect_table")
        image_table = schema.get("detected_tables", {}).get("image_table")
        label_table = schema.get("detected_tables", {}).get("label_table")

        if not defect_table:
            raise FSDServiceError(format_schema_detection_error(schema))

        file_index = build_image_file_index(img_dir if img_dir_exists else None)

        with sqlite_connection(db) as conn:
            raw_images = read_all_rows(conn, image_table)
            raw_defects = read_all_rows(conn, defect_table)
            label_map = load_label_map(conn, label_table)

        images, defects_by_image_key, defects_by_id, resolved = build_dataset(
            raw_images=raw_images,
            raw_defects=raw_defects,
            label_map=label_map,
            image_dir=img_dir if img_dir_exists else None,
            file_index=file_index,
        )

        summary = compute_summary(
            db_path=db,
            image_dir=img_dir if img_dir_exists else None,
            schema=schema,
            images=images,
            defects_by_id=defects_by_id,
            file_index=file_index,
        )

        if img_dir is not None and not img_dir_exists:
            summary["image_dir_warning"] = f"Image directory does not exist: {img_dir}"

        loaded_summary = VIEWER_STATE.set_loaded_data(
            db_path=db,
            image_dir=img_dir if img_dir_exists else None,
            schema=schema,
            label_map=label_map,
            summary=summary,
            images=images,
            defects_by_image_key=defects_by_image_key,
            defects_by_id=defects_by_id,
            resolved_image_paths=resolved,
        )

        return loaded_summary

    except FSDServiceError:
        raise
    except Exception as e:
        VIEWER_STATE.set_error(str(e))
        raise FSDServiceError(f"Failed to open FS-FSD archive: {e}") from e


def get_summary() -> Dict[str, Any]:
    """
    Return current database summary.
    """
    VIEWER_STATE.require_ready()
    return VIEWER_STATE.get_summary()


def list_images(
    *,
    class_name: Optional[str] = None,
    min_score: Optional[float] = None,
    query: Optional[str] = None,
    only_existing: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    List images with optional filters.
    """
    VIEWER_STATE.require_ready()

    images = VIEWER_STATE.list_images()
    filtered: List[Dict[str, Any]] = []

    class_filter = class_name.strip().lower() if class_name else None
    query_filter = query.strip().lower() if query else None
    min_score_value = to_float(min_score)

    for img in images:
        if only_existing and not img.get("image_exists"):
            continue

        if class_filter:
            classes = [str(c).lower() for c in img.get("classes", [])]
            if class_filter not in classes:
                continue

        if min_score_value is not None:
            max_score = to_float(img.get("max_score"))
            if max_score is None or max_score < min_score_value:
                continue

        if query_filter:
            haystack = " ".join(
                [
                    str(img.get("image_key") or ""),
                    str(img.get("image_id") or ""),
                    str(img.get("image_uid") or ""),
                    str(img.get("source_path") or ""),
                    str(img.get("resolved_path") or ""),
                ]
            ).lower()

            if query_filter not in haystack:
                continue

        filtered.append(json_safe(img))

    total = len(filtered)

    if offset < 0:
        offset = 0

    if limit is not None:
        limit = max(0, int(limit))
        filtered = filtered[offset: offset + limit]
    else:
        filtered = filtered[offset:]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "images": filtered,
    }


def record_to_public(
    record: Dict[str, Any],
    *,
    include_polygon: bool = False,
    include_coefficients: bool = False,
    polygon_points: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convert an internal defect record to public JSON-ready data.
    """
    out: Dict[str, Any] = {}

    for k, v in record.items():
        if str(k).startswith("_"):
            continue
        out[str(k)] = json_safe(v)

    if include_polygon:
        try:
            out["polygon"] = reconstruct_polygon_for_record(record, polygon_points)
            out["polygon_point_count"] = len(out["polygon"])
        except Exception as e:
            out["polygon"] = []
            out["polygon_error"] = str(e)

    if include_coefficients:
        coeffs = decode_coefficients_from_record(record)
        if coeffs is not None:
            out["fourier_coeffs"] = [
                round(float(x), 8)
                for x in coeffs.reshape(-1).tolist()
                if math.isfinite(float(x))
            ]
            out["coeff_count"] = len(out["fourier_coeffs"])
        else:
            out["fourier_coeffs"] = []
            out["coeff_count"] = out.get("coeff_count")

    return out


def get_image_records(
    image_key: str,
    *,
    polygon_points: Optional[int] = 128,
    class_name: Optional[str] = None,
    min_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Return one image plus its defect records with reconstructed polygons.
    """
    VIEWER_STATE.require_ready()

    image = VIEWER_STATE.get_image(image_key)

    if image is None:
        raise FSDServiceError(f"Image not found: {image_key}")

    records = VIEWER_STATE.get_records_for_image(image_key)

    class_filter = class_name.strip().lower() if class_name else None
    min_score_value = to_float(min_score)

    public_records: List[Dict[str, Any]] = []

    for r in records:
        if class_filter and str(r.get("class_name", "")).lower() != class_filter:
            continue

        if min_score_value is not None:
            score = to_float(r.get("score"))
            if score is None or score < min_score_value:
                continue

        public_records.append(
            record_to_public(
                r,
                include_polygon=True,
                include_coefficients=False,
                polygon_points=polygon_points,
            )
        )

    public_image = json_safe(image)
    public_image["image_url"] = f"/api/image-file?image_key={quote(str(image_key), safe='')}"

    return {
        "image": public_image,
        "records": public_records,
        "record_count": len(public_records),
    }


def get_defect(
    defect_id: str,
    *,
    include_polygon: bool = True,
    include_coefficients: bool = True,
    polygon_points: Optional[int] = 128,
) -> Dict[str, Any]:
    """
    Return one defect by defect_id.
    """
    VIEWER_STATE.require_ready()

    record = VIEWER_STATE.get_defect(str(defect_id))

    if record is None:
        raise FSDServiceError(f"Defect not found: {defect_id}")

    return record_to_public(
        record,
        include_polygon=include_polygon,
        include_coefficients=include_coefficients,
        polygon_points=polygon_points,
    )


def get_image_file_path(image_key: str) -> Path:
    """
    Resolve the image file path for FileResponse.
    """
    VIEWER_STATE.require_ready()

    image = VIEWER_STATE.get_image(image_key)

    if image is None:
        raise FSDServiceError(f"Image not found: {image_key}")

    resolved = image.get("resolved_path")

    if resolved:
        p = Path(str(resolved))
        if p.exists() and p.is_file():
            return p

    _, image_dir = VIEWER_STATE.get_paths()
    p = resolve_image_path(image, image_dir, None)

    if p is not None and p.exists() and p.is_file():
        return p

    raise FSDServiceError(
        "Image file could not be resolved. "
        f"image_key={image_key}, image_id={image.get('image_id')}, "
        f"source_path={image.get('source_path')}"
    )


def get_schema() -> Dict[str, Any]:
    """
    Return current schema snapshot.
    """
    VIEWER_STATE.require_ready()
    return VIEWER_STATE.get_schema()


def get_runtime_state() -> Dict[str, Any]:
    """
    Return current state snapshot.
    """
    return VIEWER_STATE.snapshot()