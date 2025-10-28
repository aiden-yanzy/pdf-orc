"""I/O helpers for loading manuscripts and persisting analysis artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ManuscriptIO:
    """Simple filesystem-backed helper for manuscripts and artefacts."""

    def __init__(self, *, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def read_text(self, path: Path) -> str:
        return Path(path).read_text(encoding=self.encoding)

    def write_markdown(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=self.encoding)

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding=self.encoding)

    def ensure_directory(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def ensure_run_log_directory(self, output_dir: Path) -> Path:
        return self.ensure_directory(output_dir / "run_logs")

    @staticmethod
    def infer_title(manuscript_text: str, fallback: str) -> str:
        for line in manuscript_text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            if candidate.startswith("#"):
                return candidate.lstrip("#").strip()
        return fallback
