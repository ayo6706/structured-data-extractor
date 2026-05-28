import tempfile
from pathlib import Path

import pytest

from app.core.exceptions import StorageError
from app.infrastructure.storage import LocalStorage


@pytest.fixture
def temp_storage_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
async def test_local_storage_save_load_delete(temp_storage_dir: Path) -> None:
    storage = LocalStorage(temp_storage_dir)
    file_id = "test-uuid"
    filename = "test.txt"
    content = b"hello world storage test"

    # Save
    rel_path = await storage.save(file_id, content, filename)
    assert (
        rel_path == f"{file_id}/{filename}"
        or rel_path == f"{file_id}\\{filename}"
    )

    # Load
    loaded_content = await storage.load(rel_path)
    assert loaded_content == content

    # Delete
    await storage.delete(rel_path)

    # Assert deleted file no longer exists
    with pytest.raises(StorageError) as exc_info:
        await storage.load(rel_path)
    assert "load" in str(exc_info.value)


@pytest.mark.asyncio
async def test_local_storage_save_failure() -> None:
    with tempfile.NamedTemporaryFile() as tmpfile:
        storage = LocalStorage(Path(tmpfile.name))

        with pytest.raises(StorageError) as exc_info:
            await storage.save("some-id", b"data", "test.txt")

        assert "save" in str(exc_info.value)


@pytest.mark.asyncio
async def test_local_storage_sanitizes_saved_filename(
    temp_storage_dir: Path,
) -> None:
    storage = LocalStorage(temp_storage_dir)

    rel_path = await storage.save("file-id", b"content", "../evil.pdf")

    assert rel_path == "file-id/evil.pdf" or rel_path == "file-id\\evil.pdf"
    assert (temp_storage_dir / "file-id" / "evil.pdf").exists()
    assert not (temp_storage_dir / "evil.pdf").exists()


@pytest.mark.asyncio
async def test_local_storage_rejects_path_traversal(
    temp_storage_dir: Path,
) -> None:
    storage = LocalStorage(temp_storage_dir)

    with pytest.raises(StorageError) as exc_info:
        await storage.load("../outside.pdf")

    assert "resolve" in str(exc_info.value)


@pytest.mark.asyncio
async def test_local_storage_delete_does_not_remove_storage_root(
    temp_storage_dir: Path,
) -> None:
    storage = LocalStorage(temp_storage_dir)
    root_file = temp_storage_dir / "root.txt"
    root_file.write_bytes(b"content")

    await storage.delete("root.txt")

    assert temp_storage_dir.exists()
