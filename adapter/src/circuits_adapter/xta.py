"""XTLARR01 array files (X_eTaL-gpu docs/binary-arrays.md, format version 1).

Header: 8-byte magic ``XTLARR01``, a scalar tag (1 f32, 2 f64, 3 i64), the rank
(0-9), 6 reserved zero bytes, then one little-endian u64 per dimension. The
payload follows, little-endian and row-major. Truncation and extra bytes are errors.
"""

import struct
from pathlib import Path

import numpy as np

MAGIC = b"XTLARR01"
MAX_RANK = 9
TAGS = {1: np.dtype("<f4"), 2: np.dtype("<f8"), 3: np.dtype("<i8")}
TAG_OF = {np.dtype(np.float32): 1, np.dtype(np.float64): 2, np.dtype(np.int64): 3}


class XtaError(ValueError):
    """A malformed XTLARR01 file, or an array the format cannot hold."""


def encode(a) -> bytes:
    """The array as XTLARR01 bytes."""
    a = np.asarray(a)
    tag = TAG_OF.get(a.dtype.newbyteorder("=")) if a.dtype.kind in "fi" else None
    if tag is None:
        raise XtaError(f"unsupported dtype {a.dtype}: f32, f64 or i64 only")
    if a.ndim > MAX_RANK:
        raise XtaError(f"rank {a.ndim} above {MAX_RANK}")
    header = MAGIC + bytes([tag, a.ndim]) + bytes(6) + struct.pack(f"<{a.ndim}Q", *a.shape)
    payload = np.ascontiguousarray(a, dtype=TAGS[tag]).tobytes(order="C")
    return header + payload


def _header(data: bytes) -> tuple[np.dtype, tuple[int, ...], int]:
    if len(data) < 16:
        raise XtaError("file too short for a header")
    if data[:8] != MAGIC:
        raise XtaError("bad magic: not an XTLARR01 file")
    if data[8] not in TAGS:
        raise XtaError(f"unknown scalar tag {data[8]}")
    rank = data[9]
    if rank > MAX_RANK:
        raise XtaError(f"rank {rank} above {MAX_RANK}")
    if data[10:16] != bytes(6):
        raise XtaError("reserved bytes are not zero")
    end = 16 + 8 * rank
    if len(data) < end:
        raise XtaError("file too short for its dimensions")
    return TAGS[data[8]], struct.unpack(f"<{rank}Q", data[16:end]), end


def decode(data: bytes) -> np.ndarray:
    """The array held in XTLARR01 bytes, with its stored dtype."""
    dtype, shape, start = _header(data)
    count = int(np.prod(shape, dtype=np.uint64)) if shape else 1
    if len(data) - start != count * dtype.itemsize:
        raise XtaError(f"payload length {len(data) - start} is not {count} items of {dtype}")
    return np.frombuffer(data, dtype=dtype, offset=start).reshape(shape).copy()


def write(path, a) -> None:
    """Write the array to a new file; an existing file is refused."""
    data = encode(a)
    with Path(path).open("xb") as f:
        f.write(data)


def read(path) -> np.ndarray:
    return decode(Path(path).read_bytes())


def shape(path) -> tuple[int, ...]:
    """The array's shape, from the header alone."""
    with Path(path).open("rb") as f:
        head = f.read(16)
        rank = head[9] if len(head) >= 10 else 0
        head += f.read(8 * min(rank, MAX_RANK))
    return _header(head)[1]
