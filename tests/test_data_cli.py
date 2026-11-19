"""Loading parquet shards in the release schema, and the command line, on a tiny synthetic shard."""
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

import civicbench

import _toy

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    HAVE_DATA = True
except ImportError:          # pragma: no cover - the optional extra is missing
    HAVE_DATA = False


def write_shard(path, clips, corrupt=None):
    """clips: (clip_id, frames, label, meta) -> a parquet file shaped like the release (PNG frames)."""
    rows = {"clip_id": [], "label": [], "generator": [], "frames": [], "frames_sha256": []}
    for cid, fr, lab, meta in clips:
        pngs = []
        for f in fr:
            b = io.BytesIO()
            Image.fromarray(f).save(b, format="PNG")
            pngs.append({"bytes": b.getvalue(), "path": None})
        rows["clip_id"].append(cid)
        rows["label"].append(lab)
        rows["generator"].append(meta.get("generator"))
        rows["frames"].append(pngs)
        sha = hashlib.sha256(fr.tobytes()).hexdigest()
        rows["frames_sha256"].append("0" * 64 if cid == corrupt else sha)
    pq.write_table(pa.table(rows), path)


@unittest.skipUnless(HAVE_DATA, "needs the optional extra: pip install civicbench[data]")
class TestLoad(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="civicbench_test_")
        self.clips = _toy.dataset(5, 5)
        os.makedirs(os.path.join(self.dir, "data"))
        write_shard(os.path.join(self.dir, "data", "test-00000-of-00002.parquet"), self.clips[:6])
        write_shard(os.path.join(self.dir, "data", "test-00001-of-00002.parquet"), self.clips[6:])

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_round_trip(self):
        cs = civicbench.load(self.dir, quiet=True)
        self.assertEqual(len(cs), 10)
        got = list(cs)
        self.assertEqual([c.clip_id for c in got], [c[0] for c in self.clips])
        for c, ref in zip(got, self.clips):
            np.testing.assert_array_equal(c.frames, ref[1])
            self.assertEqual(c.label, ref[2])
            self.assertEqual(c.meta["generator"], ref[3]["generator"])
        self.assertEqual(len(list(cs)), 10)                       # re-iterable
        cid, frames, label, meta = got[0]                         # unpacks as a tuple
        self.assertEqual(frames.dtype, np.uint8)

    def test_filters(self):
        self.assertEqual([c.label for c in civicbench.load(self.dir, label="generated", quiet=True)], [1] * 5)
        self.assertEqual(len(list(civicbench.load(self.dir, label="real", quiet=True))), 5)
        self.assertEqual(len(list(civicbench.load(self.dir, limit=3, quiet=True))), 3)
        one = os.path.join(self.dir, "data", "test-00001-of-00002.parquet")
        self.assertEqual(len(list(civicbench.load(one, quiet=True))), 4)

    def test_checksum(self):
        bad = os.path.join(self.dir, "bad.parquet")
        write_shard(bad, self.clips[:2], corrupt=self.clips[1][0])
        with self.assertRaises(ValueError):
            list(civicbench.load(bad, quiet=True))
        self.assertEqual(len(list(civicbench.load(bad, verify=False, quiet=True))), 2)

    def test_cli(self):
        from civicbench.cli import main
        out = os.path.join(self.dir, "res.json")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            rc = main(["evaluate", "--detector", "_toy:linear_detector", "--data", self.dir, "--budgets", "1,2,4,8,16",
                       "--attacks", "pgd,square", "--n-evasion", "2", "--n-framing", "2", "--quiet", "--out", out])
        finally:
            sys.path.pop(0)
        self.assertEqual(rc, 0)
        with open(out) as f:
            res = json.load(f)
        self.assertEqual(res["budgets"], [1, 2, 4, 8, 16])
        self.assertEqual(sorted(res["evasion_tpr"], key=int), ["1", "2", "4", "8", "16"])
        self.assertEqual(sorted(res["framing_rate"], key=int), ["1", "2", "4", "8", "16"])


class TestDefaultSource(unittest.TestCase):
    def test_default_repo_is_the_release(self):
        repos = civicbench.DEFAULT_REPO
        self.assertEqual(len(repos), 2)
        self.assertTrue(repos[0].endswith("/CivicBench-real") and repos[1].endswith("/CivicBench-generated"))

    def test_empty_source_raises(self):
        with self.assertRaises(ValueError):
            civicbench.load([])


if __name__ == "__main__":
    unittest.main()
