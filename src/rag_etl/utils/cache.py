import hashlib
import shutil

from pathlib import Path

from rag_etl.config import CONFIG


cache_path = Path(CONFIG['CACHE_DIR'])

if not cache_path.exists():
    raise ValueError(f"Cache path {cache_path} does not exist.")


def _hash_file(path: Path) -> str:
    """Return the SHA256 hex digest of the file bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_from_cache(scope: str, key_path: str, value_path: str) -> bool:
    """
    Hashes the bytes of the file `key_path`, then looks it up in the cache for the given `scope`.
    If it exists, it copies it to `value_path` and returns True.
    Otherwise, it returns False.
    """

    # If no cache for this scope, return False
    scope_path = cache_path / scope
    if not scope_path.exists():
        return False

    # Hash file
    hash = _hash_file(Path(key_path))

    # If hash not in cache, return False
    cached_file_path = scope_path / hash / Path(value_path).name
    if not cached_file_path.exists():
        return False

    # Copy file and return True
    shutil.copyfile(cached_file_path, value_path)
    return True


def set_to_cache(scope: str, key_path: str, value_path: str):
    """
    Hashes the bytes of the file `key_path`, then copies the file `value_path` to
    the cache for the given `scope`, using the hash as key.
    """

    # If no cache for this scope, create it
    scope_path = cache_path / scope
    scope_path.mkdir(parents=True, exist_ok=True)

    # Hash file
    hash = _hash_file(Path(key_path))

    # Build file path and create parent folder if needed
    cached_file_path = scope_path / hash / Path(value_path).name
    cached_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy file (overwrite if already present)
    shutil.copyfile(value_path, cached_file_path)
