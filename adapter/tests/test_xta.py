"""XTLARR01 files, per ../X_eTaL-gpu/docs/binary-arrays.md (format version 1)."""

import struct

import numpy as np
import pytest

from circuits_adapter import xta


def test_header_bytes_for_f32_matrix():
    data = xta.encode(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    assert data[:8] == b"XTLARR01"
    assert data[8] == 1  # f32
    assert data[9] == 2  # rank
    assert data[10:16] == bytes(6)
    assert struct.unpack("<QQ", data[16:32]) == (2, 3)
    assert struct.unpack("<6f", data[32:]) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


@pytest.mark.parametrize(
    ("dtype", "tag", "width"),
    [(np.float32, 1, 4), (np.float64, 2, 8), (np.int64, 3, 8)],
)
def test_scalar_tags_and_widths(dtype, tag, width):
    data = xta.encode(np.arange(5, dtype=dtype))
    assert data[8] == tag
    assert len(data) == 16 + 8 + 5 * width


def test_payload_is_little_endian_row_major():
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    data = xta.encode(np.asfortranarray(a))
    assert np.frombuffer(data[16 + 24 :], dtype="<f4").tolist() == list(range(24))


def test_round_trip_ranks_zero_to_nine():
    rng = np.random.default_rng(0)
    for rank in range(10):
        shape = tuple([2] * rank)
        a = rng.standard_normal(shape).astype(np.float32)
        b = xta.decode(xta.encode(a))
        assert b.dtype == np.float32 and b.shape == shape
        assert np.array_equal(a, b)


def test_round_trip_empty_and_scalar():
    assert xta.decode(xta.encode(np.zeros((0, 3), dtype=np.float64))).shape == (0, 3)
    s = xta.decode(xta.encode(np.float64(2.5)))
    assert s.shape == () and float(s) == 2.5


def test_refuses_unsupported_dtype_and_rank():
    with pytest.raises(xta.XtaError, match="dtype"):
        xta.encode(np.zeros(3, dtype=np.float16))
    with pytest.raises(xta.XtaError, match="rank"):
        xta.encode(np.zeros([1] * 10, dtype=np.float32))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda b: b"XTLARR02" + b[8:], "magic"),
        (lambda b: b[:8] + bytes([9]) + b[9:], "tag"),
        (lambda b: b[:9] + bytes([10]) + b[10:], "rank"),
        (lambda b: b[:10] + bytes([1]) + b[11:], "reserved"),
        (lambda b: b[:-1], "length"),
        (lambda b: b + b"\x00", "length"),
        (lambda b: b[:20], "short"),
    ],
)
def test_decode_refuses_malformed(mutate, message):
    good = xta.encode(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(xta.XtaError, match=message):
        xta.decode(mutate(good))


def test_write_refuses_existing_file(tmp_path):
    path = tmp_path / "a.xta"
    xta.write(path, np.ones(2, dtype=np.float32))
    assert np.array_equal(xta.read(path), np.ones(2, dtype=np.float32))
    with pytest.raises(FileExistsError):
        xta.write(path, np.ones(2, dtype=np.float32))


def test_shape_reads_header_only(tmp_path):
    path = tmp_path / "a.xta"
    xta.write(path, np.zeros((3, 4, 5), dtype=np.float32))
    assert xta.shape(path) == (3, 4, 5)
