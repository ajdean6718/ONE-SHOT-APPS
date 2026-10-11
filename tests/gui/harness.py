"""GUI test harness (owner: shell). Runs INSIDE a GUI Blender started with ``--enable-event-simulate``.

A scenario is a generator function ``scenario(h)``; every ``yield`` hands control back to Blender's event loop
(``yield`` = one timer tick, ``yield 1.5`` = wait 1.5 s) so simulated events are processed and the UI redraws.

    import harness

    def scenario(h):
        yield from h.wait_ready()
        yield from h.switch_workspace("Vector")
        h.check(len(h.areas("VIEW_3D")) == 1, "one 3D view in Vector")
        h.screenshot("vector")

    harness.main(scenario)

Results: ``artifacts/gui/<scenario>/result.json`` (+ screenshots ``*.png``); override the directory with
``SEAM_GUI_OUT``. Exit codes: 0 pass, 1 check failures, 2 exception, 3 timeout, 77 skipped (reason in the JSON).
"""

import json
import os
import subprocess
import sys
import threading
import time
import traceback

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXIT_PASS, EXIT_FAIL, EXIT_ERROR, EXIT_TIMEOUT, EXIT_SKIP = 0, 1, 2, 3, 77


class Skip(Exception):
    """Raise (or call ``h.skip(reason)``) to end the scenario as skipped."""


class RequireFailed(Exception):
    pass


class Harness:
    def __init__(self, name, timeout=240.0):
        self.name = name
        self.out_dir = os.environ.get("SEAM_GUI_OUT") or os.path.join(ROOT, "artifacts", "gui", name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.checks = []
        self.screenshots = []
        self.notes = []
        self.t0 = time.time()
        self.timeout = timeout
        self.status = "running"
        self._shot_no = 0
        self._watchdog = None

    # ------------------------------------------------------------------ bookkeeping
    def note(self, msg):
        print(f"[gui:{self.name}] {msg}", flush=True)
        self.notes.append(str(msg))

    def check(self, cond, msg):
        ok = bool(cond)
        self.checks.append({"ok": ok, "msg": msg})
        print(f"[gui:{self.name}] {'PASS' if ok else 'FAIL'} {msg}", flush=True)
        return ok

    def require(self, cond, msg):
        if not self.check(cond, msg):
            raise RequireFailed(msg)

    def skip(self, reason):
        raise Skip(reason)

    # ------------------------------------------------------------------ lookup
    @property
    def wm(self):
        return bpy.context.window_manager

    @property
    def window(self):
        return self.wm.windows[0]

    @property
    def screen(self):
        return self.window.screen

    def areas(self, area_type=None, workspace=None):
        """Areas of the window's current screen (or of ``workspace``'s screen), optionally of one type, left to right."""
        scr = self.screen if workspace is None else bpy.data.workspaces[workspace].screens[0]
        out = [a for a in scr.areas if area_type is None or a.type == area_type]
        return sorted(out, key=lambda a: (a.x, -a.y))

    @staticmethod
    def region(area, region_type="WINDOW"):
        return next((r for r in area.regions if r.type == region_type), None)

    def center(self, area, region_type="WINDOW"):
        r = self.region(area, region_type)
        return r.x + r.width // 2, r.y + r.height // 2

    # ------------------------------------------------------------------ waiting
    def wait(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            yield 0.05

    def wait_until(self, predicate, timeout=10.0, what="condition"):
        end = time.time() + timeout
        while time.time() < end:
            if predicate():
                return True
            yield 0.05
        self.note(f"timed out waiting for {what}")
        return False

    def wait_ready(self, seconds=2.0):
        """Wait for the window and the first redraws after startup."""
        yield from self.wait_until(lambda: len(self.wm.windows) > 0, 20, "a window")
        yield from self.wait(seconds)

    def redraw(self):
        """Force a full redraw of every area (so screenshots show current state)."""
        for win in self.wm.windows:
            for area in win.screen.areas:
                area.tag_redraw()
        try:
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        except Exception as exc:
            self.note(f"redraw_timer failed: {exc}")

    def switch_workspace(self, name, settle=1.0):
        ws = bpy.data.workspaces.get(name)
        self.require(ws is not None, f"workspace '{name}' exists")
        self.window.workspace = ws
        ok = yield from self.wait_until(lambda: self.window.workspace.name == name, 10, f"workspace {name}")
        self.require(ok, f"window switched to '{name}'")
        yield from self.wait(settle)
        self.redraw()
        yield from self.wait(0.3)

    # ------------------------------------------------------------------ input simulation
    def _ev(self, type, value, x, y, **mods):
        self.window.event_simulate(type=type, value=value, x=int(x), y=int(y), **mods)

    def mouse_move(self, x, y):
        self._ev("MOUSEMOVE", "NOTHING", x, y)

    def press(self, x, y, button="LEFTMOUSE", **mods):
        self._ev(button, "PRESS", x, y, **mods)

    def release(self, x, y, button="LEFTMOUSE", **mods):
        self._ev(button, "RELEASE", x, y, **mods)

    def click(self, x, y, button="LEFTMOUSE", **mods):
        self.mouse_move(x, y)
        yield 0.1
        self.press(x, y, button, **mods)
        yield 0.1
        self.release(x, y, button, **mods)
        yield 0.2

    def drag(self, p0, p1, steps=8, button="LEFTMOUSE", **mods):
        """Press at p0, move in ``steps`` increments, release at p1 (one tick between events)."""
        self.mouse_move(*p0)
        yield 0.15
        self.press(*p0, button, **mods)
        yield 0.15
        for i in range(1, steps + 1):
            t = i / steps
            self.mouse_move(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
            yield 0.05
        self.release(*p1, button, **mods)
        yield 0.3

    def key(self, key_type, x=None, y=None, **mods):
        """Press and release a key (mouse position matters for editor-specific keymaps)."""
        if x is None:
            x, y = self.center(self.areas("VIEW_3D")[0]) if self.areas("VIEW_3D") else (200, 200)
        self.mouse_move(x, y)
        yield 0.1
        self._ev(key_type, "PRESS", x, y, **mods)
        yield 0.05
        self._ev(key_type, "RELEASE", x, y, **mods)
        yield 0.2

    def wheel(self, x, y, up=True, **mods):
        self.mouse_move(x, y)
        yield 0.1
        self._ev("WHEELUPMOUSE" if up else "WHEELDOWNMOUSE", "PRESS", x, y, **mods)
        yield 0.2

    # ------------------------------------------------------------------ screenshots
    def screenshot(self, label):
        """Capture the whole X screen (ImageMagick ``import -window root``) into the output directory."""
        self._shot_no += 1
        path = os.path.join(self.out_dir, f"{self._shot_no:02d}_{label}.png")
        self.redraw()
        try:
            subprocess.run(["import", "-window", "root", path], check=True, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (OSError, subprocess.SubprocessError) as exc:
            self.note(f"screenshot '{label}' failed: {exc}")
            return None
        self.screenshots.append(path)
        return path

    @staticmethod
    def image_stddev(path):
        """Standard deviation (0..1) of the pixel values: ~0 means a blank/uniform capture."""
        out = subprocess.run(["convert", path, "-colorspace", "Gray", "-format", "%[fx:standard_deviation]", "info:"],
                             capture_output=True, text=True, timeout=30)
        try:
            return float(out.stdout.strip())
        except ValueError:
            return 0.0

    @staticmethod
    def region_stddev(path, x, y_top, w, h):
        """Standard deviation of a crop given in X-screen coordinates (origin top-left)."""
        out = subprocess.run(["convert", path, "-crop", f"{int(w)}x{int(h)}+{int(x)}+{int(y_top)}", "-colorspace", "Gray",
                              "-format", "%[fx:standard_deviation]", "info:"], capture_output=True, text=True, timeout=30)
        try:
            return float(out.stdout.strip())
        except ValueError:
            return 0.0

    def screen_height(self):
        return self.window.height

    # ------------------------------------------------------------------ finishing
    def result(self):
        failed = [c for c in self.checks if not c["ok"]]
        return {
            "scenario": self.name,
            "status": self.status,
            "blender": bpy.app.version_string,
            "duration_s": round(time.time() - self.t0, 2),
            "checks_total": len(self.checks),
            "checks_failed": len(failed),
            "checks": self.checks,
            "screenshots": [os.path.relpath(p, ROOT) if p.startswith(ROOT) else p for p in self.screenshots],
            "notes": self.notes,
        }

    def finish(self, status, code):
        self.status = status
        res = self.result()
        with open(os.path.join(self.out_dir, "result.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"GUI-RESULT {self.name} {status} checks={res['checks_total']} failed={res['checks_failed']}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

    def start_watchdog(self):
        def boom():
            self.note(f"timeout after {self.timeout}s")
            self.finish("timeout", EXIT_TIMEOUT)

        self._watchdog = threading.Timer(self.timeout, boom)
        self._watchdog.daemon = True
        self._watchdog.start()


def main(scenario, name=None, timeout=240.0):
    """Run ``scenario(h)`` (a generator) on Blender's timers; exits the process with the result code."""
    name = name or os.path.splitext(os.path.basename(sys.argv[sys.argv.index("--python") + 1] if "--python" in sys.argv
                                                     else "scenario"))[0]
    h = Harness(name, timeout=float(os.environ.get("SEAM_GUI_TIMEOUT", timeout)))
    h.start_watchdog()
    gen = scenario(h)

    def tick():
        try:
            n = next(gen)
        except StopIteration:
            failed = [c for c in h.checks if not c["ok"]]
            if not h.checks:
                h.note("scenario made no checks")
                h.finish("error", EXIT_ERROR)
            h.finish("fail" if failed else "pass", EXIT_FAIL if failed else EXIT_PASS)
        except Skip as exc:
            h.note(f"SKIP: {exc}")
            h.finish("skipped", EXIT_SKIP)
        except RequireFailed:
            h.finish("fail", EXIT_FAIL)
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            h.note("exception: " + tb.strip().splitlines()[-1])
            h.checks.append({"ok": False, "msg": "scenario raised: " + tb.strip().splitlines()[-1]})
            h.finish("error", EXIT_ERROR)
        if isinstance(n, (int, float)) and n > 0:
            return float(n)
        return 0.05

    bpy.app.timers.register(tick, first_interval=0.5)
