# SEAM Studio

Vector drawing, raster painting and **live 2D-to-3D links** inside Blender. SEAM is a Blender add-on
(`extensions/seam_studio`) plus an application template (`application_templates/seam`) that opens Blender
with four purpose-built workspaces:

| Workspace | What it is |
|---|---|
| **Vector** | A top-orthographic 2D canvas (rotation locked, flat shading, no floor or axes): artboards, layers, shapes, text, pathfinder, clipping, SVG/PNG export. |
| **Paint** | The same canvas in Texture Paint with Material Preview, plus an Image Editor: layered raster painting with masks and selections. |
| **Model** | An ordinary perspective 3D viewport that hides the 2D artwork, for the 3D results of your drawings. |
| **Split** | 2D canvas on the left, 3D viewport on the right: edit a shape and watch its extruded/revolved/swept result update. |

Blender's own workspaces (Layout, Modeling, Sculpting, UV Editing, Texture Paint, Shading, Animation, Rendering,
Compositing, Geometry Nodes, Scripting) stay available behind them. SEAM never replaces Blender's tools; it adds a
2D authoring layer on top of Blender's native curves, text, painting and modelling.

## Requirements

* Blender **4.0 or newer** (developed against 4.0.2 and the 5.2.2 LTS Python module). Blender itself is not bundled.
* Python packages: NumPy (Blender ships it).
* Optional, for the GUI tests and for building `startup.blend` without a display: `xvfb`, ImageMagick.
* Check everything with `python3 scripts/doctor.py`.

## Install and launch (from a checkout or the Linux tarball)

```bash
scripts/bootstrap.sh          # idempotent: checks Blender, creates the isolated profile, builds startup.blend
scripts/seam-studio           # opens Blender with the SEAM template (Vector workspace)
scripts/seam-studio my.blend  # open a file
scripts/seam-studio --background --python-expr "import bpy; print(bpy.data.workspaces.keys())"   # headless smoke
```

* The profile lives in `.seam-profile/` (config + scripts), separate from your normal Blender settings. Override with
  `SEAM_PROFILE=/path`.
* `SEAM_BLENDER=/path/to/blender` selects the Blender executable (default: `blender` on `PATH`).
* `scripts/bootstrap.sh --rebuild-template` rebuilds `startup.blend` (workspace layout) after the add-on changed.
  `scripts/build_template.py` documents exactly what the build does and is safe to re-run.

### Install into your own Blender instead

`python3 scripts/package.py` writes `dist/`: `seam_studio-<v>-addon.zip` (Preferences > Add-ons > Install),
`seam_studio-<v>-extension.zip` (Blender 4.2+ extensions), `seam-template-<v>.zip` (File > New > Install Application
Template, then pick *Seam*), the `seam-studio-<v>-linux.tar.gz` bundle and `SHA256SUMS`. The template enables the
add-on when it starts.

## Workflows

1. **Start panel** (sidebar tab *SEAM*, shown while the document is empty): *New Vector Document*, *New Paint Canvas*,
   *Draw to 3D*. A button that cannot work in your build is disabled and says why.
2. **Vector**: draw shapes with the toolbar tools, style them in the sidebar, arrange layers and artboards.
3. **Make 3D**: select a closed shape, choose a recipe (extrude, revolve, sweep, 3D text). The result stays live: edit the
   2D nodes and the 3D mesh follows. Pause, detach (bake) or branch it from the *Live Link* panel.
4. **Paint**: raster layers with opacity, blend modes and masks; native brushes; selections drive the stencil mask.
5. **Document panel**: units and DPI, a global *Live Updates* switch (also in the 3D view header), engine statistics,
   *Health Check* (missing link sources, missing or unpacked images, missing fonts, orphaned derived objects),
   *Repair SEAM Workspaces*.

### Shortcuts (3D View; rebind in Preferences > Keymap)

| Keys | Action |
|---|---|
| Ctrl Alt 1 / 2 / 3 / 4 | Switch to Vector / Paint / Model / Split |
| Ctrl Alt L | Pause / resume live updates |
| Ctrl Alt 0 | Fit the active artboard (when the vector tools are installed) |
| Ctrl Alt J / Ctrl Alt Shift J | Group / ungroup (when the vector tools are installed) |

They are verified not to collide with Blender's default 3D View or global keymaps (`keymaps.find_conflicts()`).

### Preferences (Preferences > Add-ons > SEAM Studio)

Show the Start panel, default artboard size, default units and DPI for new documents.

## Tests

```bash
scripts/test.sh unit        # pure-Python kernels (src/seam_core)
scripts/test.sh int52       # integration on Blender 5.2.2 LTS (bpy module, headless)
scripts/test.sh int40       # integration on Blender 4.0.2 (binary, headless)
scripts/test.sh gui         # real GUI under Xvfb: tests/gui/run_all.sh (screenshots in artifacts/gui/<scenario>/)
```

GUI scenarios start the app template with `--enable-event-simulate`, switch workspaces, send simulated mouse/keyboard
events and take screenshots; each scenario writes `result.json` and exits non-zero on failure. Logs of every run are in
`artifacts/`.

## Repository layout

```
src/seam_core/                pure Python kernels (no bpy)
extensions/seam_studio/       the add-on (seam_core is a symlink here; scripts/package.py copies it)
  ui/                         preferences, panels, workspaces, keymaps, health/migration, quick starts
application_templates/seam/   app template (startup.blend is generated, splash.png is procedural)
scripts/                      bootstrap, launcher, doctor, package, build_template, build_demos, make_splash, test.sh
tests/{unit,integration,gui}  test suites          resources/demos/   demo documents (built headlessly)
docs/                         architecture.md (design + contracts), acceptance.md (status of every claim)
```

## Known gaps

See `docs/acceptance.md` for what is tested, native, unverified or blocked in this environment. Notable limits:
the interactive GUI was verified on Blender 4.0.2 (the only interactive build obtainable here); 5.2.2 LTS is verified
headless only. Raster layers are 8-bit (Blender cannot round-trip packed float images). Boolean results are polygonal.

## License

GPL-3.0-or-later, see `LICENSE`. Third-party notes: `THIRD_PARTY_NOTICES.md`.
