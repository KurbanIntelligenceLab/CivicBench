"""The progress display: quiet, compact in logs, one live line on a terminal."""
import io
import unittest

from civicbench._progress import Progress, block_bar, fmt_time


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


def _run(steps, dt, **kw):
    clock, out = Clock(), io.StringIO()
    p = Progress(steps, "test", unit="clips", stream=out, clock=clock, **kw)
    for i in range(steps):
        clock.t += dt
        p.update(1, status=f"evasion @2 · pgd", extra=f"broken {i // 3}/{i + 1}")
    p.close()
    return out.getvalue().splitlines()


class TestProgress(unittest.TestCase):
    def test_log_mode_is_coarse(self):
        for dt in (0.01, 1.0, 10.0, 100.0):
            lines = _run(1000, dt)
            self.assertLessEqual(len(lines), 11, f"{dt} s per step: {len(lines)} lines")
        lines = _run(1000, 10.0)                   # 10,000 s: one line per 10% and a summary
        self.assertEqual(len(lines), 10)
        self.assertTrue(lines[-1].startswith("test: 1000/1000 clips"))
        self.assertIn("10%", lines[0])
        self.assertEqual(len(_run(1000, 1.0)), 4)  # 1,000 s: every 5 min is rarer than every 10%

    def test_quick_steps_print_nothing(self):
        self.assertEqual(_run(1000, 0.001), [])    # 1 s in all: below the 2 s delay

    def test_quiet_prints_nothing(self):
        for dt in (0.01, 10.0):
            self.assertEqual(_run(1000, dt, quiet=True), [])

    def test_nested_progress_is_silent(self):
        clock, out = Clock(), io.StringIO()
        with Progress(10, "outer", stream=out, clock=clock):
            inner = Progress(10, "inner", stream=out, clock=clock)
            self.assertTrue(inner.quiet)
            inner.close()
        after = Progress(10, "again", stream=out, clock=clock)
        self.assertFalse(after.quiet)
        after.close()

    def test_tty_line(self):
        clock, out = Clock(), FakeTTY()
        p = Progress(40, "evaluate", unit="clips", stream=out, clock=clock, backend="plain", refresh=3600)
        clock.t = 12.0
        p.update(18, status="framing @4 · square", extra="broken 3/12")
        line = p.line(width=200)
        for part in ("▕", "▏", "45%", "18/40 clips", "0:12 elapsed", "ETA", "framing @4 · square", "broken 3/12"):
            self.assertIn(part, line)
        self.assertLessEqual(len(p.line(width=40).replace("\033[36m", "").replace("\033[0m", "")), 40)
        p.close()
        self.assertTrue(out.getvalue().endswith("\n"))             # one summary line (the step lasted 12 s)
        self.assertEqual(out.getvalue().count("\n"), 1)

    def test_helpers(self):
        self.assertEqual(block_bar(0.0, 10), "▕" + " " * 10 + "▏")
        self.assertEqual(block_bar(1.0, 10), "▕" + "█" * 10 + "▏")
        self.assertEqual(len(block_bar(0.37, 10)), 12)
        self.assertEqual(fmt_time(75), "1:15")
        self.assertEqual(fmt_time(3725), "1:02:05")
        self.assertEqual(fmt_time(None), "--:--")


if __name__ == "__main__":
    unittest.main()
