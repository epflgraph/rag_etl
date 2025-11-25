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
    If it exists, it copies the file or folder to `value_path` and returns True.
    Otherwise, it returns False.
    """

    # If no cache for this scope, return False
    scope_path = cache_path / scope
    if not scope_path.exists():
        return False

    # Hash file
    hash = _hash_file(Path(key_path))

    # If hash folder not in cache, return False
    cache_hash_path = scope_path / hash
    if not cache_hash_path.exists():
        return False

    # Build paths
    value_path = Path(value_path)
    cache_resource_path = cache_hash_path / value_path.name

    # If no resource in cache, return False
    if not cache_resource_path.exists():
        return False

    # Copy file(s) from cache_path to value_path
    if cache_resource_path.is_dir():
        # Assuming value_path should be a folder. We create it and copy all files (non-recursive)
        value_path.mkdir(parents=True, exist_ok=True)
        for item in cache_resource_path.iterdir():
            if item.is_file():
                shutil.copyfile(item, value_path / item.name)

        return True
    elif cache_resource_path.is_file():
        # Assuming value_path should be a file. Copy file (overwrite if already present)
        value_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_resource_path, value_path)

        return True
    else:
        return False


def set_to_cache(scope: str, key_path: str, value_path: str):
    """
    Hashes the bytes of the file `key_path`, then copies the file or folder `value_path` to
    the cache for the given `scope`, using the hash as key.
    """

    # If no cache for this scope, create it
    scope_path = cache_path / scope
    scope_path.mkdir(parents=True, exist_ok=True)

    # Hash file
    hash = _hash_file(Path(key_path))

    # Build cache hash path and create parent folder if needed
    cache_hash_path = scope_path / hash
    cache_hash_path.mkdir(parents=True, exist_ok=True)

    # Build paths
    value_path = Path(value_path)
    cache_resource_path = cache_hash_path / value_path.name

    # Copy file(s) from value_path to cache_path
    if value_path.is_dir():
        # Assuming cache_resource_path should be a folder. We create it and copy all files (non-recursive)
        cache_resource_path.mkdir(parents=True, exist_ok=True)
        for item in value_path.iterdir():
            if item.is_file():
                shutil.copyfile(item, cache_resource_path / item.name)
    elif value_path.is_file():
        # Assuming value_path should be a file. Copy file (overwrite if already present)
        shutil.copyfile(value_path, cache_resource_path)
