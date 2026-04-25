# -*- coding: utf-8 -*-
"""
FSD WebGL Damage Intelligence Viewer - Runtime State
====================================================

This module stores the current local viewer session state.

It is intentionally lightweight and dependency-free. The service layer
(`fsd_service.py`) is responsible for reading the SQLite database and then
pushing normalized images, defects, schema, and summary objects into this state.

Design goals
------------
- Keep the current database session in memory.
- Avoid keeping a SQLite connection open across FastAPI/Uvicorn threads.
- Provide thread-safe read/write access.
- Store internal data required for polygon reconstruction.
- Return defensive copies to API handlers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple


def now_local_iso() -> str:
    """
    Return local time in a stable ISO-like format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_deepcopy(value: Any) -> Any:
    """
    Best-effort deepcopy.

    Most values in the state are JSON-like dictionaries/lists, but a few internal
    fields may contain bytes or numpy arrays. copy.deepcopy normally supports
    these, but this helper keeps the state methods robust.
    """
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


@dataclass
class ViewerState:
    """
    Thread-safe state container for the local WebGL viewer session.
    """

    db_path: Optional[Path] = None
    image_dir: Optional[Path] = None
    opened_at: Optional[str] = None
    session_id: int = 0

    schema: Dict[str, Any] = field(default_factory=dict)
    label_map: Dict[str, str] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    images: List[Dict[str, Any]] = field(default_factory=list)
    images_by_key: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    defects_by_image_key: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    defects_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    resolved_image_paths: Dict[str, str] = field(default_factory=dict)

    last_error: Optional[str] = None
    logs: List[str] = field(default_factory=list)

    _lock: RLock = field(default_factory=RLock, repr=False)

    # ------------------------------------------------------------------
    # Basic session lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """
        Reset the viewer session.
        """
        with self._lock:
            self.db_path = None
            self.image_dir = None
            self.opened_at = None
            self.session_id += 1

            self.schema = {}
            self.label_map = {}
            self.summary = {}

            self.images = []
            self.images_by_key = {}

            self.defects_by_image_key = {}
            self.defects_by_id = {}

            self.resolved_image_paths = {}

            self.last_error = None
            self.logs = []
            self._append_log_unlocked("Session reset.")

    def is_ready(self) -> bool:
        """
        Return True if a database session is currently loaded.
        """
        with self._lock:
            return self.db_path is not None and bool(self.summary)

    def require_ready(self) -> None:
        """
        Raise RuntimeError if no database session is loaded.
        """
        if not self.is_ready():
            raise RuntimeError(
                "No FSD database session is loaded. Please open a database first."
            )

    # ------------------------------------------------------------------
    # Logging and errors
    # ------------------------------------------------------------------
    def _append_log_unlocked(self, message: str) -> None:
        line = f"[{now_local_iso()}] {message}"
        self.logs.append(line)

        # Keep the in-memory log small.
        if len(self.logs) > 300:
            self.logs = self.logs[-300:]

    def append_log(self, message: str) -> None:
        """
        Add a runtime log line.
        """
        with self._lock:
            self._append_log_unlocked(message)

    def set_error(self, message: str) -> None:
        """
        Store the latest error message.
        """
        with self._lock:
            self.last_error = str(message)
            self._append_log_unlocked(f"ERROR: {message}")

    def clear_error(self) -> None:
        """
        Clear the latest error.
        """
        with self._lock:
            self.last_error = None

    # ------------------------------------------------------------------
    # Write loaded data
    # ------------------------------------------------------------------
    def set_loaded_data(
        self,
        *,
        db_path: Path,
        image_dir: Optional[Path],
        schema: Dict[str, Any],
        label_map: Dict[str, str],
        summary: Dict[str, Any],
        images: List[Dict[str, Any]],
        defects_by_image_key: Dict[str, List[Dict[str, Any]]],
        defects_by_id: Dict[str, Dict[str, Any]],
        resolved_image_paths: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Replace the current session with a fully loaded database session.
        """
        with self._lock:
            self.session_id += 1
            self.db_path = Path(db_path)
            self.image_dir = Path(image_dir) if image_dir is not None else None
            self.opened_at = now_local_iso()

            self.schema = safe_deepcopy(schema or {})
            self.label_map = safe_deepcopy(label_map or {})

            self.images = safe_deepcopy(images or [])
            self.images_by_key = {
                str(img.get("image_key")): img
                for img in self.images
                if img.get("image_key") is not None
            }

            self.defects_by_image_key = safe_deepcopy(defects_by_image_key or {})
            self.defects_by_id = safe_deepcopy(defects_by_id or {})
            self.resolved_image_paths = safe_deepcopy(resolved_image_paths or {})

            normalized_summary = safe_deepcopy(summary or {})
            normalized_summary["session_id"] = self.session_id
            normalized_summary["opened_at"] = self.opened_at
            normalized_summary["db_path"] = str(self.db_path)
            normalized_summary["image_dir"] = str(self.image_dir) if self.image_dir else None

            self.summary = normalized_summary
            self.last_error = None

            self._append_log_unlocked(
                "Database session loaded: "
                f"{normalized_summary.get('image_count', 0)} images, "
                f"{normalized_summary.get('defect_count', 0)} defects."
            )

            return safe_deepcopy(self.summary)

    # ------------------------------------------------------------------
    # Read state snapshots
    # ------------------------------------------------------------------
    def get_paths(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Return current database path and image directory.
        """
        with self._lock:
            return (
                Path(self.db_path) if self.db_path else None,
                Path(self.image_dir) if self.image_dir else None,
            )

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a copy of the current summary.
        """
        with self._lock:
            return safe_deepcopy(self.summary)

    def get_schema(self) -> Dict[str, Any]:
        """
        Return a copy of the current SQLite schema summary.
        """
        with self._lock:
            return safe_deepcopy(self.schema)

    def get_label_map(self) -> Dict[str, str]:
        """
        Return a copy of the current label map.
        """
        with self._lock:
            return safe_deepcopy(self.label_map)

    def list_images(self) -> List[Dict[str, Any]]:
        """
        Return all normalized image entries.
        """
        with self._lock:
            return safe_deepcopy(self.images)

    def get_image(self, image_key: str) -> Optional[Dict[str, Any]]:
        """
        Return one image entry by image_key.
        """
        with self._lock:
            img = self.images_by_key.get(str(image_key))
            return safe_deepcopy(img) if img is not None else None

    def get_records_for_image(self, image_key: str) -> List[Dict[str, Any]]:
        """
        Return all defect records for an image.
        """
        with self._lock:
            records = self.defects_by_image_key.get(str(image_key), [])
            return safe_deepcopy(records)

    def get_defect(self, defect_id: str) -> Optional[Dict[str, Any]]:
        """
        Return one defect record by defect_id.
        """
        with self._lock:
            record = self.defects_by_id.get(str(defect_id))
            return safe_deepcopy(record) if record is not None else None

    def get_logs(self) -> List[str]:
        """
        Return current runtime logs.
        """
        with self._lock:
            return list(self.logs)

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a compact state snapshot for diagnostics.
        """
        with self._lock:
            return {
                "ready": self.db_path is not None and bool(self.summary),
                "session_id": self.session_id,
                "db_path": str(self.db_path) if self.db_path else None,
                "image_dir": str(self.image_dir) if self.image_dir else None,
                "opened_at": self.opened_at,
                "image_count": len(self.images),
                "defect_count": len(self.defects_by_id),
                "last_error": self.last_error,
                "summary": safe_deepcopy(self.summary),
            }


# Global singleton used by the local viewer backend.
VIEWER_STATE = ViewerState()

# Short alias if needed by future modules.
STATE = VIEWER_STATE