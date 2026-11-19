"""Load CivicBench clips from a Hugging Face dataset repo or from local parquet shards.

Each parquet row holds one clip: ``frames`` (a list of 11 lossless PNG images, 512 x 512 RGB, the
official test input), ``label`` (0 = real, 1 = generated) and metadata columns (e.g. ``clip_id``,
``scene_kind``, ``source``, ``generator``, ``frames_sha256``). ``load`` yields ``Clip`` tuples
``(clip_id, frames, label, meta)`` with ``frames`` as uint8 ``[11, 512, 512, 3]``.

Needs the optional extra ``civicbench[data]`` (pyarrow and Pillow for local shards; ``datasets`` for
a Hugging Face repo id).
"""
from __future__ import annotations

import glob
import hashlib
import io
import os
from typing import Iterator, NamedTuple, Optional, Sequence, Union

import numpy as np

from civicbench._progress import Progress

# CivicBench on Hugging Face: one open dataset with the real and the generated clips.
DEFAULT_REPO: str = "kurbanintelligencelab/CivicBench"


class Clip(NamedTuple):
    clip_id: str
    frames: np.ndarray       # uint8 [T, H, W, 3]
    label: int               # 0 = real, 1 = generated
    meta: dict


def _label(v) -> int:
    if isinstance(v, str):
        return 0 if v.lower() == "real" else 1
    return int(v)


def _decode_image(v) -> np.ndarray:
    from PIL import Image
    if isinstance(v, dict):
        if v.get("bytes") is not None:
            img = Image.open(io.BytesIO(v["bytes"]))
        elif v.get("path"):
            img = Image.open(v["path"])
        else:
            raise ValueError("an image entry holds neither bytes nor a path")
    elif isinstance(v, (bytes, bytearray)):
        img = Image.open(io.BytesIO(v))
    elif hasattr(v, "convert"):
        img = v
    else:
        raise TypeError(f"cannot decode a frame of type {type(v).__name__}")
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _to_clip(row: dict, verify: bool) -> Clip:
    frames = np.ascontiguousarray(np.stack([_decode_image(f) for f in row["frames"]]), dtype=np.uint8)
    meta = {k: v for k, v in row.items() if k not in ("frames", "label", "clip_id")}
    sha = meta.get("frames_sha256")
    if verify and sha:
        got = hashlib.sha256(frames.tobytes()).hexdigest()
        if got != sha:
            raise ValueError(f"clip {row.get('clip_id')}: decoded frames do not match frames_sha256")
    return Clip(str(row.get("clip_id")), frames, _label(row["label"]), meta)


def _parquet_files(path: str, split: Optional[str]) -> list:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    if split:
        mine = [f for f in files if os.path.basename(f).startswith(split + "-") or os.path.basename(f) == split + ".parquet"
                or f"{os.sep}{split}{os.sep}" in f]
        files = mine or files
    if not files:
        raise FileNotFoundError(f"no parquet files under {path}")
    return files


class ClipSet:
    """A re-iterable view of one or more CivicBench sources; every iteration re-reads them in order."""

    def __init__(self, sources: Sequence[str], split: Optional[str], label, limit: Optional[int], verify: bool,
                 batch_size: int, quiet: bool):
        self.sources, self.split, self.limit, self.verify = list(sources), split, limit, verify
        self.batch_size, self.quiet = batch_size, quiet
        self.label = None if label is None else _label(label)

    def _local(self, path) -> Iterator[dict]:
        import pyarrow.parquet as pq
        for f in _parquet_files(path, self.split):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=self.batch_size):
                labels = batch.column("label").to_pylist() if "label" in batch.schema.names else None
                rows = batch.to_pylist() if labels is None or self.label is None else \
                    [r for r, lab in zip(batch.to_pylist(), labels) if _label(lab) == self.label]
                yield from rows

    def _hub(self, repo: str) -> Iterator[dict]:
        try:
            import datasets
        except ImportError as e:            # pragma: no cover - depends on the environment
            raise ImportError("loading from a Hugging Face repo needs `pip install civicbench[data]`") from e
        yield from datasets.load_dataset(repo, split=self.split or "test", streaming=True)

    def _n_rows(self) -> Optional[int]:
        try:
            import pyarrow.parquet as pq
            n = 0
            for src in self.sources:
                if not os.path.exists(src):
                    return None
                n += sum(pq.ParquetFile(f).metadata.num_rows for f in _parquet_files(src, self.split))
            return n if self.label is None else None
        except Exception:
            return None

    def __len__(self) -> int:
        n = self._n_rows()
        if n is None:
            raise TypeError("the number of clips is not known without reading them (remote source or label filter)")
        return min(n, self.limit) if self.limit is not None else n

    def __iter__(self) -> Iterator[Clip]:
        n_rows = self._n_rows()
        total = min(n_rows, self.limit) if (n_rows is not None and self.limit is not None) else (n_rows or self.limit)
        k = 0
        prog = Progress(total, "loading", unit="clips", quiet=self.quiet)
        try:
            for src in self.sources:
                rows = self._local(src) if os.path.exists(src) else self._hub(src)
                for row in rows:
                    if self.label is not None and _label(row["label"]) != self.label:
                        continue
                    if self.limit is not None and k >= self.limit:
                        return
                    clip = _to_clip(row, self.verify)
                    k += 1
                    prog.update(1, status=clip.clip_id)
                    yield clip
        finally:
            prog.close()


def load(source: Union[str, Sequence[str], None] = None, split: Optional[str] = "test", label=None,
         limit: Optional[int] = None, verify: bool = True, batch_size: int = 8, quiet: bool = False) -> ClipSet:
    """CivicBench clips from a Hugging Face repo id (e.g. ``"<org>/<name>"``), a local directory of
    parquet shards, a single parquet file, or a list of these (read in order, e.g. the real and the
    generated repo).

    split     shard prefix / split name (``test``: files ``test-*.parquet``); local directories without
              matching files use all their parquet files.
    label     keep only "real"/0 or "generated"/1 clips.
    limit     stop after this many clips.
    verify    check every clip's ``frames_sha256`` (when the column exists) after decoding.
    quiet     no progress line while iterating.
    Returns a re-iterable ``ClipSet`` of ``Clip(clip_id, frames, label, meta)``.
    """
    if source is None:
        source = DEFAULT_REPO
    sources = [source] if isinstance(source, (str, os.PathLike)) else list(source)
    if not sources:
        raise ValueError("No CivicBench source given: pass a Hugging Face repo id or a local directory of "
                         "parquet shards, e.g. civicbench.load('path/to/civicbench/data').")
    return ClipSet([os.fspath(s) for s in sources], split, label, limit, verify, batch_size, quiet)


__all__ = ["load", "Clip", "ClipSet", "DEFAULT_REPO"]
