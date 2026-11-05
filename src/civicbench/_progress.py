"""One quiet, live progress line for long steps (load, attack, evaluate, CLI).

* On a terminal: ONE line redrawn in place, shown only once a step has lasted ``delay`` seconds
  (default 2), with a spinner, a block bar, percent, done/total, elapsed time, ETA and a short
  live status. It ends with one summary line. ``rich`` draws it when importable (extra
  ``civicbench[fancy]``); otherwise a built-in fallback does, with no extra dependency.
* Not a terminal (e.g. cluster logs): one compact line at coarse milestones only, every 10% of
  the work or every ``log_every`` seconds (default 300), whichever is rarer, then the summary line.
* ``quiet=True`` prints nothing. ``NO_COLOR`` (any value) turns colors off.
* While a progress line is live, progress objects created inside it (an attack inside evaluate,
  the clip loader it reads from) stay silent, so there is never more than one line.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import sys
import threading
import time
from typing import Optional

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
EIGHTHS = " ▏▎▍▌▋▊▉"
_ACTIVE = [0]                      # live progress objects (nesting depth)
_LOCK = threading.Lock()


def fmt_time(s: Optional[float]) -> str:
    if s is None or not math.isfinite(s):
        return "--:--"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def block_bar(frac: float, width: int = 20) -> str:
    """``▕████▌     ▏``: a bar of ``width`` cells with eighth-cell resolution."""
    frac = min(max(frac, 0.0), 1.0)
    eighths = int(round(frac * width * 8))
    full, part = divmod(eighths, 8)
    body = "█" * full + (EIGHTHS[part] if part and full < width else "")
    return "▕" + body.ljust(width) + "▏"


def _rich_available() -> bool:
    try:
        import rich.live  # noqa: F401
        return True
    except Exception:
        return False


class Progress:
    """``with Progress(total, "evaluate", unit="clips") as p: ... p.update(1, status="...")``."""

    def __init__(self, total: Optional[int], desc: str = "", unit: str = "steps", quiet: bool = False,
                 stream=None, delay: float = 2.0, log_every: float = 300.0, backend: Optional[str] = None,
                 clock=time.monotonic, refresh: float = 0.1):
        self.total, self.desc, self.unit = (int(total) if total is not None else None), desc, unit
        self.stream = stream if stream is not None else sys.stdout
        self.delay, self.log_every, self.clock, self.refresh = delay, log_every, clock, refresh
        self.done, self.status, self.extra = 0, "", ""
        self.t0 = clock()
        self.nested = _ACTIVE[0] > 0
        self.quiet = quiet or self.nested
        tty = bool(getattr(self.stream, "isatty", lambda: False)())
        if backend is None:
            backend = ("rich" if _rich_available() else "plain") if tty else "log"
        self.backend = backend
        self.color = "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
        self.shown, self.closed, self._frame = False, False, 0
        self._last_log_t, self._last_log_frac = self.t0, 0.0
        self._rich = self._task = None
        self._thread = None
        self._stop = threading.Event()
        if not self.quiet:
            with _LOCK:
                _ACTIVE[0] += 1
            if self.backend in ("plain", "rich"):
                self._thread = threading.Thread(target=self._tick, daemon=True)
                self._thread.start()

    # ---- public ---------------------------------------------------------------------------
    def update(self, n: int = 1, status: Optional[str] = None, extra: Optional[str] = None) -> None:
        self.done += n
        if status is not None:
            self.status = status
        if extra is not None:
            self.extra = extra
        if self.quiet:
            return
        if self.backend == "log":
            self._maybe_log()
        elif self.backend == "rich" and self._rich is not None:
            self._rich_update()

    def set_status(self, status: str, extra: Optional[str] = None) -> None:
        self.update(0, status=status, extra=extra)

    def close(self, summary: Optional[str] = None, force: bool = False) -> None:
        """Remove the live line and print one summary line (if the step was long enough, or ``force``)."""
        if self.closed:
            return
        self.closed = True
        if self.quiet:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with _LOCK:
            _ACTIVE[0] -= 1
        elapsed = self.clock() - self.t0
        if summary is None:
            summary = (f"{self.desc}: {self.done}" + (f"/{self.total}" if self.total is not None else "")
                       + f" {self.unit} in {fmt_time(elapsed)}" + (f" · {self.extra}" if self.extra else ""))
        long_enough = force or self.shown or elapsed >= self.delay
        if self.backend == "rich":
            if self._rich is not None:
                self._rich.stop()
            if long_enough:
                self._console().print(summary, highlight=False, markup=False)
        elif self.backend == "plain":
            if self.shown:
                self.stream.write("\r\033[K")
            if long_enough:
                self.stream.write(summary + "\n")
            self.stream.flush()
        elif long_enough:
            self.stream.write(summary + "\n")
            self.stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ---- rendering ------------------------------------------------------------------------
    def _frac(self) -> Optional[float]:
        return min(self.done / self.total, 1.0) if self.total else None

    def _eta(self, elapsed: float) -> Optional[float]:
        if not self.total or self.done <= 0:
            return None
        return elapsed / self.done * max(self.total - self.done, 0)

    def line(self, width: Optional[int] = None) -> str:
        width = width or shutil.get_terminal_size((100, 20)).columns
        wide = width >= 110                    # narrow terminals get a shorter bar and time field
        elapsed = self.clock() - self.t0
        frac = self._frac()
        spin = SPINNER[self._frame % len(SPINNER)]
        parts = [f"{spin} {self.desc}"]
        if frac is not None:
            bar = block_bar(frac, 20 if wide else 12)
            if self.color:
                bar = "\033[36m" + bar + "\033[0m"
            parts.append(f"{bar} {100 * frac:3.0f}% {self.done}/{self.total} {self.unit}")
        else:
            parts.append(f"{self.done} {self.unit}")
        eta = fmt_time(self._eta(elapsed))
        parts.append(f"{fmt_time(elapsed)} elapsed · ETA {eta}" if wide else f"{fmt_time(elapsed)} · ETA {eta}")
        if self.status:
            parts.append(self.status)
        if self.extra:
            parts.append(self.extra)
        text = " · ".join(parts[:1]) + " " + " · ".join(parts[1:])
        return _truncate(text, max(width - 1, 20))

    def _tick(self) -> None:
        while not self._stop.wait(self.refresh):
            if self.clock() - self.t0 < self.delay:
                continue
            if self.backend == "rich":
                if self._rich is None:
                    self._rich_start()
                else:
                    self._rich_update()
                continue
            self._frame += 1
            self.shown = True
            try:
                self.stream.write("\r" + self.line() + "\033[K")
                self.stream.flush()
            except Exception:
                return

    def _maybe_log(self) -> None:
        now = self.clock()
        if now - self._last_log_t < max(self.log_every, self.delay):
            return
        frac = self._frac()
        if frac is not None:
            if frac >= 1.0 or frac < self._last_log_frac + 0.1 - 1e-12:
                return
            self._last_log_frac = math.floor(frac * 10 + 1e-9) / 10
        self._last_log_t = now
        elapsed = now - self.t0
        pct = f" {100 * frac:.0f}%" if frac is not None else ""
        tot = f"/{self.total}" if self.total is not None else ""
        msg = (f"[{self.desc}]{pct} {self.done}{tot} {self.unit} · {fmt_time(elapsed)} elapsed · "
               f"ETA {fmt_time(self._eta(elapsed))}" + (f" · {self.status}" if self.status else "")
               + (f" · {self.extra}" if self.extra else ""))
        self.shown = True
        self.stream.write(msg + "\n")
        self.stream.flush()

    # ---- rich backend -----------------------------------------------------------------------
    def _console(self):
        from rich.console import Console
        if not hasattr(self, "_con"):
            self._con = Console(file=self.stream, no_color="NO_COLOR" in os.environ)
        return self._con

    def _rich_start(self) -> None:
        """rich draws the same line through a transient Live display (cursor handling, resize, NO_COLOR)."""
        from rich.live import Live
        from rich.text import Text
        con = self._console()

        def render():
            self._frame += 1
            return Text.from_ansi(self.line(width=con.width), no_wrap=True, overflow="ellipsis")
        self._rich = Live(get_renderable=render, console=con, transient=True, refresh_per_second=1 / self.refresh)
        self._rich.start()
        self.shown = True

    def _rich_update(self) -> None:
        pass                                   # Live redraws on its own refresh thread


_ANSI = re.compile(r"\033\[[0-9;]*m")


def _truncate(text: str, width: int) -> str:
    """Cut ``text`` to at most ``width`` visible characters (ANSI color codes do not count)."""
    if len(_ANSI.sub("", text)) <= width:
        return text
    out, vis, pos = [], 0, 0
    for m in _ANSI.finditer(text):
        chunk = text[pos:m.start()]
        take = chunk[:max(width - 1 - vis, 0)]
        out.append(take)
        vis += len(take)
        out.append(m.group())
        pos = m.end()
    tail = text[pos:]
    out.append(tail[:max(width - 1 - vis, 0)])
    return "".join(out) + "…" + ("\033[0m" if _ANSI.search(text) else "")


__all__ = ["Progress", "block_bar", "fmt_time"]
