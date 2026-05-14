"""Download only the MAESTRO test-split audio files (~13 GB, 178 tracks).

Google no longer serves individual audio files; this script reads the ZIP central
directory via HTTP range requests and extracts only the test-split entries.

Usage:
    python scripts/download_maestro_test.py --data-home data/raw/maestro
    python scripts/download_maestro_test.py --data-home data/raw/maestro --max-tracks 20
"""

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

VERSION = "v3.0.0"
BASE_URL = f"https://storage.googleapis.com/magentadata/datasets/maestro/{VERSION}"
ZIP_URL = f"{BASE_URL}/maestro-{VERSION}.zip"


class _HTTPRangeReader(io.RawIOBase):
    """Seekable read-only wrapper over an HTTP URL using byte-range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._pos = 0
        req = urllib.request.Request(url, method="HEAD")
        resp = urllib.request.urlopen(req)
        self._size = int(resp.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def readinto(self, b: bytearray) -> int:
        n = len(b)
        end = min(self._pos + n - 1, self._size - 1)
        if self._pos > end:
            return 0
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self._pos}-{end}"}
        )
        data = urllib.request.urlopen(req, timeout=60).read()
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)


def _load_meta(data_home: Path) -> list[dict]:
    """Download v3 metadata if needed; return list of track dicts."""
    meta_path = data_home / f"maestro-{VERSION}.json"
    if not meta_path.exists():
        url = f"{BASE_URL}/maestro-{VERSION}.json"
        print(f"Fetching metadata from {url} …")
        data_home.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, meta_path)

    with open(meta_path) as f:
        raw = json.load(f)

    # v3 uses a columnar dict {field: {idx: value}}; v2 used a list of dicts
    if isinstance(raw, list):
        return raw
    indices = list(raw["split"].keys())
    return [{k: raw[k][i] for k in raw} for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-home", default="data/raw/maestro")
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Download only this many test tracks (for a quick subset)",
    )
    args = parser.parse_args()

    data_home = Path(args.data_home)
    data_home.mkdir(parents=True, exist_ok=True)

    meta = _load_meta(data_home)
    test_tracks = [v for v in meta if v["split"] == "test"]
    if args.max_tracks:
        test_tracks = test_tracks[: args.max_tracks]

    needed = [t for t in test_tracks if not (data_home / t["audio_filename"]).exists()]
    already = len(test_tracks) - len(needed)
    if already:
        print(f"{already} tracks already on disk, skipping.")
    if not needed:
        print("All done — nothing to download.")
        return

    # ZIP entries are stored as "maestro-v3.0.0/<year>/<file>.wav"
    ZIP_PREFIX = f"maestro-{VERSION}/"
    needed_set = {ZIP_PREFIX + t["audio_filename"] for t in needed}
    total_dur = sum(t["duration"] for t in needed)
    print(
        f"Downloading {len(needed)} test tracks ({total_dur/3600:.1f}h audio) "
        f"via range requests on {ZIP_URL} …"
    )

    print("Opening ZIP central directory (2-3 range requests) …")
    reader = _HTTPRangeReader(ZIP_URL)
    zf = zipfile.ZipFile(io.BufferedReader(reader))

    done = 0
    for info in zf.infolist():
        name = info.filename
        if name not in needed_set:
            continue
        # Strip the ZIP prefix so the file lands at data_home/<year>/<file>
        rel_path = name[len(ZIP_PREFIX):]
        dest = data_home / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        done += 1
        print(f"  [{done}/{len(needed)}] {name}", end="", flush=True)
        with zf.open(info) as src, open(dest, "wb") as dst:
            size = 0
            buf = bytearray(1 << 20)  # 1 MB chunks
            while True:
                n = src.readinto(buf)
                if not n:
                    break
                dst.write(buf[:n])
                size += n
        print(f"  {size/1e6:.0f} MB")

    zf.close()
    print(f"\nDone ({done}/{len(needed)} downloaded).")


if __name__ == "__main__":
    main()
