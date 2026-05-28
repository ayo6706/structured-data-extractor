import os
from pathlib import Path
from typing import Protocol

import anyio

from app.core.exceptions import StorageError


class StorageBackend(Protocol):
    async def save(self, file_id: str, content: bytes, filename: str) -> str:
        ...

    async def load(self, file_path: str) -> bytes:
        ...

    async def delete(self, file_path: str) -> None:
        ...


class LocalStorage:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    async def save(self, file_id: str, content: bytes, filename: str) -> str:
        safe_file_id = self._safe_name(file_id, fallback="file")
        safe_filename = self._safe_name(filename, fallback="uploaded.pdf")
        target_dir = self._resolve_relative_path(safe_file_id)
        relative_path = Path(safe_file_id) / safe_filename
        target_path = self._resolve_relative_path(str(relative_path))

        def _save_sync() -> None:
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(content)
            except Exception as exc:
                raise StorageError(
                    path=str(target_path),
                    operation="save",
                    original_exception=exc,
                ) from exc

        try:
            await anyio.to_thread.run_sync(_save_sync)
            return str(relative_path)
        except Exception as exc:
            if isinstance(exc, StorageError):
                raise
            raise StorageError(
                path=str(target_path),
                operation="save",
                original_exception=exc,
            ) from exc

    async def load(self, file_path: str) -> bytes:
        full_path = self._resolve_relative_path(file_path)

        def _load_sync() -> bytes:
            try:
                return full_path.read_bytes()
            except Exception as exc:
                raise StorageError(
                    path=str(full_path),
                    operation="load",
                    original_exception=exc,
                ) from exc

        try:
            return await anyio.to_thread.run_sync(_load_sync)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                path=str(full_path),
                operation="load",
                original_exception=exc,
            ) from exc

    async def delete(self, file_path: str) -> None:
        full_path = self._resolve_relative_path(file_path)

        def _delete_sync() -> None:
            try:
                if full_path.exists():
                    full_path.unlink()
                parent_dir = full_path.parent
                if (
                    parent_dir.resolve() != self.base_dir.resolve()
                    and parent_dir.exists()
                    and not os.listdir(parent_dir)
                ):
                    parent_dir.rmdir()
            except Exception as exc:
                raise StorageError(
                    path=str(full_path),
                    operation="delete",
                    original_exception=exc,
                ) from exc

        try:
            await anyio.to_thread.run_sync(_delete_sync)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                path=str(full_path),
                operation="delete",
                original_exception=exc,
            ) from exc

    def _resolve_relative_path(self, file_path: str) -> Path:
        try:
            relative_path = Path(file_path)
            if relative_path.is_absolute():
                raise ValueError("storage path must be relative")

            base_path = self.base_dir.resolve()
            full_path = (base_path / relative_path).resolve()
            full_path.relative_to(base_path)
            return full_path
        except Exception as exc:
            raise StorageError(
                path=file_path,
                operation="resolve",
                original_exception=exc,
            ) from exc

    @staticmethod
    def _safe_name(value: str, *, fallback: str) -> str:
        name = Path(value.replace("\\", "/")).name
        if name in {"", ".", ".."}:
            return fallback
        return name
