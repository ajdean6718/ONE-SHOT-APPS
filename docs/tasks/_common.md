# Common rules for every SEAM worker (read once)

- Read `CLAUDE.md` and `docs/architecture.md` (§ relevant to you) first. Contracts = signatures/docstrings in the stub files.
- Work ONLY in your worktree directory and ONLY on your owned paths. If you need a change elsewhere, write it in your handoff ("REQUEST: ...") instead.
- Runtimes: Blender 4.0.2 (`blender`, GUI+headless) and official bpy 5.2.2 LTS (`/opt/seam-cache/venv-bpy52/bin/python`). Code must pass on both. Do not install or download Blender/bpy again.
- Tests: `scripts/test.sh unit`, `scripts/test.sh int52`, `scripts/test.sh int40` (pass extra pytest args after the suite, e.g. `scripts/test.sh int52 -k link`). Integration fixtures: `seam` (fresh empty file, add-on enabled), `engine`, `blend_path`; helpers in `tests/integration/helpers.py` (`save_and_reload`, `flush`, `make_curve`). Background mode has no timers: call `engine.flush()` after edits.
- Verified facts (don't re-research): edit-mode curve splines are live in `curve.splines`; `Object.to_mesh()` works on an unlinked curve object; `bpy.ops.ed.undo()` works headless; `delaunay_2d_cdt(output_type=1, need_ids=True)` gives overlap provenance; 8-bit packed images round-trip exactly, float ones do not; `Window.event_simulate` exists (needs `--enable-event-simulate`); Cycles + Workbench available in both runtimes; `Paint.brush` read-only in 5.2; unified paint settings location differs (use `compat.unified_paint_settings`).
- Never mutate bpy data from threads. Never push undo steps from producers. Use `poll_message_set` for disabled operators. No fake success messages.
- Keep heavy local processes modest: the machine has 4 CPUs shared with other workers. Run focused test subsets while iterating; run your full owned suites on both runtimes before finishing.
- Commit early and often on your branch with clear messages (end each commit message with the two attribution lines below). Do not push. Do not merge other branches.
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Bxa6DuA9Tma8YVfmJMbd3D
- Final answer (your return value) = compact handoff: branch + last commit hash, changed paths, exact test commands + pass/fail counts per runtime, known gaps/blockers, REQUESTs for the integrator, what this unblocks. Full logs stay in `artifacts/`.
