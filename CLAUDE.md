# SEAM Studio — operational map

Blender-native 2D vector + raster + 3D creative app with live 2D→3D links.
Authoritative design/contract: `docs/architecture.md`. Checkpoint: `docs/state.json`.
Effective workflow config + measured run data: `docs/execution.md`. Acceptance: `docs/acceptance.md`.

## Layout
- `src/seam_core/` — pure Python (+numpy; `mathutils` only in `boolean2d`). No `bpy`.
- `extensions/seam_studio/` — the Blender add-on. `seam_core` is a symlink here in dev, copied by `scripts/package.py`.
- `application_templates/seam/` — app template (workspaces in generated `startup.blend`).
- `tests/unit` (pure), `tests/integration` (real bpy, headless), `tests/gui` (real GUI under Xvfb).
- `scripts/` — bootstrap, doctor, launch, test, package. `artifacts/` and `dist/` are ignored outputs.

## Runtimes (do not reinstall)
- GUI + headless: Ubuntu `blender` 4.0.2 (`/usr/bin/blender`, Python 3.12, numpy 1.26). Official download hosts are egress-blocked.
- Headless LTS target: official `bpy==5.2.2` (5.2.2 LTS) venv at `/opt/seam-cache/venv-bpy52` (Python 3.13).
- Code must work on both. Version differences go through `extensions/seam_studio/compat.py`.

## Commands
- Unit: `scripts/test.sh unit` · Integration: `scripts/test.sh int40` / `int52` · GUI: `scripts/test.sh gui` · All: `scripts/test.sh all`
- Launch: `scripts/seam-studio [file.blend]` (isolated profile in `.seam-profile/`).

## Rules for workers
- Edit only the paths your task packet owns. Shared files (`__init__.py`, `props.py`, `compat.py`, `engine/`, manifests) belong to the integrator; request changes in your handoff.
- Never mutate `bpy.data` off the main thread. Engine builds run in timers / `engine.flush()`.
- Every enabled control must do something real; use `poll_message_set` to explain disabled ones.
- Tests must drive real operators. Never report a test as passing without running it; keep full logs in `artifacts/`.
- Handoff = changed paths, commit hash, commands run + results, blockers, what it unblocks. Keep it short.
