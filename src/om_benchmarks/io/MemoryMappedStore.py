import mmap
from io import BufferedReader
from pathlib import Path
from typing import Dict, Tuple

from zarr.abc.store import ByteRequest, OffsetByteRequest, RangeByteRequest
from zarr.core.buffer import BufferPrototype, default_buffer_prototype
from zarr.core.buffer.core import Buffer
from zarr.storage import LocalStore


class MemoryMappedStore(LocalStore):
    """
    Read-only memory-mapped file system store for Zarr, with mmap caching.
    """

    def __init__(self, root: Path | str) -> None:
        super().__init__(root=root, read_only=True)
        self._mmap_cache: Dict[str, Tuple[mmap.mmap, BufferedReader]] = {}

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if prototype is None:
            prototype = default_buffer_prototype()
        if not self._is_open:
            await self._open()
        path = self.root / key

        # Don't mmap known metadata files
        if key.endswith((".zarray", ".zgroup", ".zattrs")):
            return await super().get(key, prototype, byte_range)

        try:
            if key not in self._mmap_cache:
                f = open(path, "rb")
                mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
                self._mmap_cache[key] = (mm, f)
            else:
                mm, _ = self._mmap_cache[key]
            if byte_range is not None:
                if isinstance(byte_range, RangeByteRequest):
                    start = byte_range.start
                    stop = byte_range.end
                elif isinstance(byte_range, OffsetByteRequest):
                    start = byte_range.offset
                    stop = mm.__len__()
                else:
                    start = mm.__len__() - byte_range.suffix
                    stop = mm.__len__()
                data = mm[start:stop]
            else:
                data = mm[:]
            return prototype.buffer.from_bytes(data)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    def close(self):
        errors = []
        for mm, f in self._mmap_cache.values():
            try:
                mm.close()
            except Exception as e:
                errors.append(e)
            try:
                f.close()
            except Exception as e:
                errors.append(e)
        self._mmap_cache.clear()
        if errors:
            raise RuntimeError(f"Errors occurred during close: {errors}")
