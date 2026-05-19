"""Generate placeholder PNGs (logo + favicon) for kalmanbox docs.

Produces tiny solid-colour RGB PNGs without any non-stdlib dependency
so the documentation build works on every CI environment.  Replace
with real renders of ``docs/assets/images/logo.svg`` once the final
artwork is approved.

Usage::

    python scripts/make_placeholder_pngs.py
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "assets" / "images"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def make_solid_png(path: Path, size: int, rgb: tuple[int, int, int]) -> None:
    """Write a square solid-colour RGB PNG."""
    width = height = size
    pixel = bytes(rgb)
    raw = bytearray()
    for _ in range(height):
        raw.append(0x00)  # filter byte: None
        raw += pixel * width
    compressed = zlib.compress(bytes(raw), level=9)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    print(f"wrote {path.relative_to(ROOT)} ({len(png)} bytes)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # kalmanbox brand indigo (#2c3e8a)
    make_solid_png(OUT_DIR / "logo.png", 256, (44, 62, 138))
    make_solid_png(OUT_DIR / "favicon.png", 64, (44, 62, 138))


if __name__ == "__main__":
    main()
