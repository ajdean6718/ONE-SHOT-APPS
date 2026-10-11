"""SEAM derived-data engine (owner: integrator). Contract: docs/architecture.md §5.

One depsgraph handler maps changed IDs to dirty keys ("<kind>:<id>") via registered
producers. Builds run on the main thread (timer tick in the GUI, ``flush()`` in
background/tests). A build is skipped when the producer's current signature equals
the stored one, which makes rebuilds idempotent and loop-free and lets undo/redo/load
simply revalidate everything. Producers may split work into prepare (main thread) /
compute (pure, worker thread) / apply (main thread); stale async results are dropped.
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import bpy
from bpy.app.handlers import persistent


@dataclass
class BuildResult:
    key: str
    status: str  # "ok" | "skipped" | "error" | "conflict" | "waiting" | "obsolete" | "stale"
    message: str = ""
    ms: float = 0.0


class Producer:
    """Base class for derived-artifact producers. Subclass and register with ``register_producer``."""

    kind: str = ""
    priority: int = 50          # lower builds first within a flush pass
    async_capable: bool = False

    def keys_for_update(self, scene, id_orig, geometry: bool, transform: bool) -> Iterable[str]:
        return ()

    def all_keys(self, scene) -> Iterable[str]:
        return ()

    def signature(self, scene, key: str) -> Optional[str]:
        raise NotImplementedError

    def stored_signature(self, scene, key: str) -> Optional[str]:
        raise NotImplementedError

    def build(self, scene, key: str, signature: str) -> BuildResult:
        raise NotImplementedError

    # optional async split -------------------------------------------------
    def prepare(self, scene, key: str):
        raise NotImplementedError

    def compute(self, snapshot):
        raise NotImplementedError

    def apply(self, scene, key: str, signature: str, result) -> BuildResult:
        raise NotImplementedError

    def wants_async(self, snapshot) -> bool:
        return False


_producers: Dict[str, Producer] = {}
_dirty: Dict[str, None] = {}
_update_hooks: List[Callable] = []
_revalidate = False
_building = False
_suspend = 0
_pending: Dict[str, tuple] = {}  # key -> (signature, future)
_executor: Optional[ThreadPoolExecutor] = None
_stats = {"flushes": 0, "builds": 0, "skipped": 0, "errors": 0, "last_flush_ms": 0.0,
          "max_build_ms": 0.0, "last_build_ms": 0.0, "by_kind": {}, "stale_dropped": 0}
_last_slow_ms = 0.0
_generation = 0   # bumped whenever scene data may have changed outside the engine's control
_indexing = 0     # > 0 while the engine maps updates to keys or runs a build loop

MAX_PASSES = 4


# ---------------------------------------------------------------- registry

def register_producer(producer: Producer) -> None:
    if not producer.kind:
        raise ValueError("producer.kind must be set")
    _producers[producer.kind] = producer


def unregister_producer(kind: str) -> None:
    _producers.pop(kind, None)


def producers() -> Dict[str, Producer]:
    return dict(_producers)


def add_update_hook(fn: Callable) -> None:
    """``fn(scene, depsgraph)`` runs at the start of every depsgraph update (e.g. id dedupe, adoption)."""
    if fn not in _update_hooks:
        _update_hooks.append(fn)


def remove_update_hook(fn: Callable) -> None:
    if fn in _update_hooks:
        _update_hooks.remove(fn)


# ---------------------------------------------------------------- dirty tracking

def mark_dirty(key: str) -> None:
    _dirty[key] = None
    _schedule()


def request_revalidate() -> None:
    """Re-check every key's signature on the next flush (cheap: unchanged keys are skipped)."""
    global _revalidate
    _revalidate = True
    _schedule()


def pending_keys() -> List[str]:
    return list(_dirty)


def is_building() -> bool:
    return _building


def index_generation() -> Optional[int]:
    """A token for per-pass lookup caches (owner -> derived objects, parent -> children, ...).

    Not None only while the engine maps a depsgraph update to keys or runs a build pass; it changes whenever
    data may have changed behind the engine's back (each depsgraph update, each flush pass), so a cache built
    under one token is valid for that whole handler call / pass. Producers that create or delete objects
    during a pass must update their own cache entries. Outside the engine it is None: compute directly."""
    return _generation if _indexing else None


def _bump_generation() -> None:
    global _generation
    _generation += 1


@contextmanager
def suspend():
    """Ignore depsgraph updates while active (bulk edits that are rebuilt explicitly)."""
    global _suspend
    _suspend += 1
    try:
        yield
    finally:
        _suspend -= 1


def _settings(scene):
    seam = getattr(scene, "seam", None)
    return getattr(seam, "engine", None) if seam is not None else None


def _live_enabled(scene) -> bool:
    s = _settings(scene)
    return True if s is None else bool(s.live_enabled)


# ---------------------------------------------------------------- handlers

@persistent
def _on_depsgraph_update(scene, depsgraph):
    global _indexing
    _bump_generation()
    if _building or _suspend:
        return
    for hook in list(_update_hooks):
        try:
            hook(scene, depsgraph)
        except Exception:  # a hook must never break the depsgraph callback
            traceback.print_exc()
    if not _producers:
        return
    _bump_generation()  # hooks may have changed ids / parents
    added = False
    _indexing += 1
    try:
        for upd in depsgraph.updates:
            idd = upd.id
            orig = getattr(idd, "original", idd)
            geom = bool(upd.is_updated_geometry)
            xform = bool(upd.is_updated_transform)
            for p in _producers.values():
                try:
                    for key in p.keys_for_update(scene, orig, geom, xform):
                        _dirty[key] = None
                        added = True
                except ReferenceError:
                    continue
                except Exception:
                    traceback.print_exc()
    finally:
        _indexing -= 1
    if added:
        _schedule()


@persistent
def _on_undo_redo(scene, *_args):
    _bump_generation()
    request_revalidate()


@persistent
def _on_load_post(*_args):
    _bump_generation()
    _dirty.clear()
    _pending.clear()
    request_revalidate()


@persistent
def _on_save_pre(*_args):
    try:
        flush()
    except Exception:
        traceback.print_exc()


_HANDLERS = (
    ("depsgraph_update_post", _on_depsgraph_update),
    ("undo_post", _on_undo_redo),
    ("redo_post", _on_undo_redo),
    ("load_post", _on_load_post),
    ("save_pre", _on_save_pre),
)


# ---------------------------------------------------------------- scheduling

def _delay_seconds() -> float:
    try:
        s = _settings(bpy.context.scene)
        base = (s.debounce_ms if s is not None else 0.0) / 1000.0
    except Exception:
        base = 0.0
    # adaptive: if builds are slow, coalesce more aggressively (max 250 ms)
    if _last_slow_ms > 30.0:
        base = max(base, min(0.25, 2.0 * _last_slow_ms / 1000.0))
    return base


def _schedule() -> None:
    if bpy.app.background:
        return  # tests and scripts call flush() explicitly
    try:
        if not bpy.app.timers.is_registered(_tick):
            bpy.app.timers.register(_tick, first_interval=_delay_seconds())
    except Exception:
        pass


def _tick():
    try:
        scene = bpy.context.scene
    except Exception:
        scene = None
    if scene is None:
        return 0.1 if (_dirty or _revalidate or _pending) else None
    try:
        flush(scene, evaluate=False)
    except Exception:
        traceback.print_exc()
    if _pending:
        return 0.02
    if _dirty or _revalidate:
        return _delay_seconds() or 0.01
    return None


# ---------------------------------------------------------------- flush

def _sorted_keys(keys: Iterable[str]) -> List[str]:
    def prio(k):
        p = _producers.get(k.split(":", 1)[0])
        return (p.priority if p else 999, k)
    return sorted(set(keys), key=prio)


def _executor_get() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="seam-compute")
    return _executor


def _record(res: BuildResult) -> None:
    global _last_slow_ms
    st = _stats
    kind = res.key.split(":", 1)[0]
    k = st["by_kind"].setdefault(kind, {"builds": 0, "ms_total": 0.0, "ms_max": 0.0, "errors": 0})
    if res.status == "ok":
        st["builds"] += 1
        k["builds"] += 1
        k["ms_total"] += res.ms
        k["ms_max"] = max(k["ms_max"], res.ms)
        st["max_build_ms"] = max(st["max_build_ms"], res.ms)
        st["last_build_ms"] = res.ms
        _last_slow_ms = res.ms if res.ms > 30.0 else _last_slow_ms * 0.5
    elif res.status == "skipped":
        st["skipped"] += 1
    elif res.status in ("error",):
        st["errors"] += 1
        k["errors"] += 1


def _collect_pending(scene, results: List[BuildResult]) -> None:
    for key, (sig, fut) in list(_pending.items()):
        if not fut.done():
            continue
        _pending.pop(key, None)
        p = _producers.get(key.split(":", 1)[0])
        if p is None:
            continue
        try:
            current = p.signature(scene, key)
        except Exception:
            current = None
        if current != sig:
            _stats["stale_dropped"] += 1
            _dirty[key] = None  # rebuild from the newest state
            results.append(BuildResult(key, "stale", "superseded by a newer edit"))
            continue
        t0 = time.perf_counter()
        try:
            res = p.apply(scene, key, sig, fut.result())
        except Exception as exc:
            traceback.print_exc()
            res = BuildResult(key, "error", f"{type(exc).__name__}: {exc}")
        res.ms = res.ms or (time.perf_counter() - t0) * 1000.0
        _record(res)
        results.append(res)


def _build_one(scene, key: str, results: List[BuildResult], sync_async: bool) -> None:
    p = _producers.get(key.split(":", 1)[0])
    if p is None:
        return
    try:
        sig = p.signature(scene, key)
    except ReferenceError:
        return
    except Exception as exc:
        traceback.print_exc()
        res = BuildResult(key, "error", f"signature failed: {type(exc).__name__}: {exc}")
        _record(res)
        results.append(res)
        return
    if sig is None:
        results.append(BuildResult(key, "obsolete"))
        return
    if key in _pending and _pending[key][0] == sig:
        return  # identical computation already running
    if sig == p.stored_signature(scene, key):
        res = BuildResult(key, "skipped")
        _record(res)
        results.append(res)
        return
    t0 = time.perf_counter()
    try:
        if p.async_capable:
            snap = p.prepare(scene, key)
            if not sync_async and p.wants_async(snap):
                _pending[key] = (sig, _executor_get().submit(p.compute, snap))
                results.append(BuildResult(key, "waiting"))
                return
            res = p.apply(scene, key, sig, p.compute(snap))
        else:
            res = p.build(scene, key, sig)
    except Exception as exc:
        traceback.print_exc()
        res = BuildResult(key, "error", f"{type(exc).__name__}: {exc}")
    res.ms = res.ms or (time.perf_counter() - t0) * 1000.0
    _record(res)
    results.append(res)


def flush(scene=None, evaluate: Optional[bool] = None, wait_async: Optional[bool] = None) -> List[BuildResult]:
    """Process dirty keys now. Returns the build results of this call.

    ``evaluate`` (default: True in background) runs ``view_layer.update()`` before and
    between passes so depsgraph handlers fire for pending data changes. ``wait_async``
    (default: True in background) computes async producers synchronously.
    """
    global _revalidate, _building, _indexing
    scene = scene or bpy.context.scene
    background = bpy.app.background
    evaluate = background if evaluate is None else evaluate
    wait_async = background if wait_async is None else wait_async
    results: List[BuildResult] = []
    if not _live_enabled(scene):
        return results
    t_start = time.perf_counter()
    for _pass in range(MAX_PASSES):
        if evaluate:
            try:
                bpy.context.view_layer.update()
            except Exception:
                pass
        _bump_generation()
        _collect_pending(scene, results)
        if _revalidate:
            _revalidate = False
            _indexing += 1
            try:
                for p in list(_producers.values()):
                    try:
                        for key in p.all_keys(scene):
                            _dirty[key] = None
                    except Exception:
                        traceback.print_exc()
            finally:
                _indexing -= 1
        if not _dirty:
            break
        keys = _sorted_keys(_dirty)
        _dirty.clear()
        _building = True
        _indexing += 1
        try:
            for key in keys:
                _build_one(scene, key, results, sync_async=wait_async)
        finally:
            _building = False
            _indexing -= 1
        if not evaluate:
            break  # GUI: the next depsgraph update reschedules follow-up passes
    _stats["flushes"] += 1
    _stats["last_flush_ms"] = (time.perf_counter() - t_start) * 1000.0
    return results


def stats() -> dict:
    out = dict(_stats)
    out["by_kind"] = {k: dict(v) for k, v in _stats["by_kind"].items()}
    out["pending"] = len(_pending)
    out["dirty"] = len(_dirty)
    return out


def reset() -> None:
    """Forget dirty/pending state and statistics (tests, file load)."""
    global _revalidate, _last_slow_ms
    _dirty.clear()
    _pending.clear()
    _revalidate = False
    _last_slow_ms = 0.0
    _stats.update({"flushes": 0, "builds": 0, "skipped": 0, "errors": 0, "last_flush_ms": 0.0,
                   "max_build_ms": 0.0, "last_build_ms": 0.0, "by_kind": {}, "stale_dropped": 0})


def register():
    for name, fn in _HANDLERS:
        lst = getattr(bpy.app.handlers, name)
        if fn not in lst:
            lst.append(fn)


def unregister():
    for name, fn in _HANDLERS:
        lst = getattr(bpy.app.handlers, name)
        while fn in lst:
            lst.remove(fn)
    try:
        if bpy.app.timers.is_registered(_tick):
            bpy.app.timers.unregister(_tick)
    except Exception:
        pass
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None
    reset()
