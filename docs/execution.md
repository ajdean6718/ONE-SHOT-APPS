# Execution record (effective configuration and measured run data)

## Effective Claude workflow configuration (replaced 2026-10-11)
Backup of previous files: `/root/.claude/backups/seam-config-20261011/` (container-local) and git history
(`git show 54f5b4c:.claude/settings.json`). Rollback: copy `project-settings.json` back to `.claude/settings.json`.

Changes in `.claude/settings.json` (take effect at the next session start; this session keeps its startup snapshot):
- `workflowSizeGuideline: "unrestricted"`, `worktree.baseRef: "head"`, `env.CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH: "2"`.
- Plugins trimmed to build-relevant ones (feature-dev, superpowers, wshobson python-development, unit-testing, tdd-workflows,
  debugging-toolkit, error-debugging, code-refactoring, application-performance, comprehensive-review, documentation-generation,
  accessibility-compliance, developer-essentials). ECC (~46k always-on tokens) and 15 web/backend/payment plugins disabled.
- Hooks: kept secret-scanner, config-protection, spawn-gate, artifact-verify, check-settings-secrets, check-session-state,
  session-end, cost-tracker. Removed per-call/low-value hooks: loop-detector (every tool call), gate-guard (warn-only),
  track-edits + batch-check, write-evidence (HTML/PDF deliverables only), startup-sync (times out: GitHub blocked).
- `CLAUDE.md` created as the short operational map.

## Verified environment facts
- 4 vCPU, 15 GiB RAM, no GPU (llvmpipe GL 4.5 under Xvfb), Ubuntu 24.04.
- Egress: `download.blender.org`, GitHub, Blender mirrors, flathub, snapcraft → blocked (403/000). PyPI, Ubuntu archive, conda-forge, npm → allowed.
- Only `claude-plugins-official` plugins are actually installed; wshobson/ECC marketplaces never installed (GitHub blocked).
- Host env sets `CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH=1` for this session → delegation is flattened through the director.
- Workflow concurrency cap observed in docs: min(16, CPUs−2) = 2 agents per workflow → several workflows run side by side.

## Deviations from the brief
- Official Blender binary unobtainable (egress policy). GUI runtime = Ubuntu Blender 4.0.2; LTS target 5.2.2 verified headless only.
- wshobson specialist agents unavailable as registered agents; workers are general workflow agents given the relevant task packet + skills.
- Raster layers are 8-bit only (float packed images do not round-trip in Blender: measured).

## Model routing (this run)
- Director: claude-opus-5-5 (session model). Link engine/critical path: Opus (high effort) — hardest data-integrity work.
- Implementation workers: Sonnet 5.5, high effort. Reviews: Sonnet 5.5 high (critical areas get an Opus review). Mechanical checks: scripts.

## Measured run data
(appended as the run progresses)

### Wave 1 (started 00:43 UTC)
| Workflow | Streams | Wall time | Subagent tokens | Agents |
|---|---|---|---|---|
| wdut7olpa | core-geometry, core-raster-io | 55.2 min | 1,100,052 | 5 (review passed core-raster-io → no fix stage) |
| wtt3zsr22 | link (Opus), vector-a | 48.2 min | 1,429,473 | 6 |
| wtz8vd0i5 | vector-b | 43.8 min | 723,202 | 3 |
| wi938n4vn | paint, shell | 87.4 min | 1,628,647 | 6 |
Observed concurrency: 7 implementation agents at once (4 workflows), load average 3.2–4.5 on 4 vCPU.
Integration defects found at merge: 1 (gradient-image purge by name prefix deleted foreign images → ownership marker), fixed in ac00f1e.
Link update latency (from link worker, under load avg ~4.5): 4-pt square 1.4 ms (4.0.2) / 2.5 ms (5.2.2) median; 200-pt curved shape 37–38 ms median.

### Wave 2 (started 01:48 UTC)
| Workflow | Streams | Wall time | Subagent tokens | Agents |
|---|---|---|---|---|
| w5p54pewk | connect, compositing | 29.8 min | 873,496 | 5 (compositing review passed) |
| wzcgoec8u | acceptance | (running) | | |
| wur1zyh4x | gui | (running) | | |
| wfsvtolbj | integrated review (4 lenses + per-finding refutation) | (running) | | |
Integration defects found at merge: gui_draw_rect held a Scene reference across memfile undo (test bug) — fixed e36cd93.
Integrated build after wave-2 merges (b3d40e7): unit 485 (py3.13) / 453+1 skip (py3.12); integration 515+1 skip (5.2.2 LTS), 514+2 skip (4.0.2);
GUI 5/5 scenarios (4.0.2 under Xvfb). Template rebuild 47 s; demos 3/3 built.
