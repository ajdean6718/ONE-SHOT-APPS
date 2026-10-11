# SEAM Studio performance (measured)

Every number below was measured by `scripts/bench.py` on the machine described here and copied from `artifacts/perf/summary.json`; nothing is extrapolated. Times are wall-clock milliseconds, shown as **median / p95** over the stated number of repetitions. Regenerate with `python3 scripts/bench.py --all`.

## Machine and runtimes

|  | value |
| --- | --- |
| CPU | Intel(R) Xeon(R) Processor @ 2.80GHz (4 logical CPUs, 4 usable) |
| Memory | 15.72 GiB |
| OS | Linux-6.18.44-fc-v114-x86_64-with-glibc2.39 |
| Repository commit | f7a09a9 |
| Generated | 2026-10-11T02:39:01+0000 |
| blender-4.0.2 | blender binary (background), 4.0.2, Python 3.12.3, numpy 1.26.4, add-on enable 216 ms; load average at start/end [3.48, 2.77, 2.6] / [5.35, 4.61, 3.46] |
| bpy-5.2.2 | bpy module, 5.2.2 LTS, Python 3.13.16, numpy 2.5.4, add-on enable 54 ms; load average at start/end [1.73, 2.12, 2.42] / [3.48, 2.77, 2.6] |

The machine has 4 CPUs that were shared with other processes while measuring (see the load averages); treat small differences between runtimes as noise.

## Link update latency

An edit moves one source node (as a Grab in Edit Mode does) and the engine is flushed. *flush after edit* is the whole `engine.flush()` call (it includes the `view_layer.update()` that the background engine performs, i.e. depsgraph evaluation of the scene); *engine build* is the build time the engine reports for the link only. Repetitions per case: see `edits` in the JSON (blender-4.0.2: 40, bpy-5.2.2: 40).

### median / p95 ms

|  | blender-4.0.2 | bpy-5.2.2 |
| --- | --- | --- |
| 4-point square (extrude) - flush after edit | 1.86 / 1.96 | 1.84 / 2.23 |
| 4-point square (extrude) - engine build | 1.07 / 1.15 | 1.11 / 1.32 |
| 4-point square (extrude) - result size | 16 vertices, 8 faces | 16 vertices, 8 faces |
| 200-point polygon (extrude) - flush after edit | 9.00 / 9.31 | 8.92 / 10.15 |
| 200-point polygon (extrude) - engine build | 6.47 / 6.88 | 7.20 / 8.19 |
| 200-point polygon (extrude) - result size | 800 vertices, 596 faces | 800 vertices, 596 faces |
| text 'SEAM Studio' (Text 3D) - flush after edit | 16.29 / 20.20 | 17.09 / 19.27 |
| text 'SEAM Studio' (Text 3D) - engine build | 13.01 / 14.65 | 14.03 / 15.42 |
| text 'SEAM Studio' (Text 3D) - result size | 2976 vertices, 2192 faces | 2976 vertices, 2192 faces |
| 8-point profile, 32 segments (revolve) - flush after edit | 3.53 / 4.17 | 4.24 / 5.57 |
| 8-point profile, 32 segments (revolve) - engine build | 2.52 / 2.98 | 3.16 / 4.23 |
| 8-point profile, 32 segments (revolve) - result size | 194 vertices, 224 faces | 194 vertices, 224 faces |
| 12-node open path (sweep) - flush after edit | 7.56 / 8.14 | 8.51 / 12.61 |
| 12-node open path (sweep) - engine build | 5.84 / 6.16 | 7.32 / 11.00 |
| 12-node open path (sweep) - result size | 1456 vertices, 1436 faces | 1456 vertices, 1436 faces |

## Document size scaling

Shapes are fills (every 2nd also stroked, every 5th with a gradient) on a 1920 x 1080 artboard, no links. *flush, no change* is `engine.flush()` with nothing dirty; *flush, one node edit* edits one shape and flushes; *node edit + view_layer.update only* is the same edit followed by just the depsgraph evaluation (no engine builds): the share of a flush that Blender itself spends.

### median / p95 ms unless stated

|  | blender-4.0.2 | bpy-5.2.2 |
| --- | --- | --- |
| 10 shapes - flush, no change | 0.00 / 0.01 | 0.00 / 0.01 |
| 10 shapes - flush, one node edit | 3.64 / 9.28 | 4.13 / 9.76 |
| 10 shapes - node edit + view_layer.update only | 0.25 / 0.36 | 0.27 / 0.40 |
| 10 shapes - builds per edit (median) | 1.0 | 1.0 |
| 10 shapes - first flush after creation (ms) | 34.1 | 109.0 |
| 10 shapes - save / open .blend (ms) | 1.4 / 13.3 | 7.2 / 12.3 |
| 10 shapes - .blend size (bytes) | 733860 | 113481 |
| 100 shapes - flush, no change | 0.00 / 0.00 | 0.00 / 0.01 |
| 100 shapes - flush, one node edit | 4.49 / 4.74 | 6.95 / 12.55 |
| 100 shapes - node edit + view_layer.update only | 1.20 / 1.34 | 1.60 / 2.11 |
| 100 shapes - builds per edit (median) | 1.0 | 1.0 |
| 100 shapes - first flush after creation (ms) | 498.6 | 589.3 |
| 100 shapes - save / open .blend (ms) | 4.1 / 36.3 | 20.6 / 68.0 |
| 100 shapes - .blend size (bytes) | 3599468 | 287555 |
| 500 shapes - flush, no change | 0.00 / 0.01 | 0.00 / 0.01 |
| 500 shapes - flush, one node edit | 25.38 / 34.50 | 1028.61 / 1138.15 |
| 500 shapes - node edit + view_layer.update only | 9.69 / 15.82 | 17.71 / 21.49 |
| 500 shapes - builds per edit (median) | 2.0 | 2.0 |
| 500 shapes - first flush after creation (ms) | 3400.9 | 6678.9 |
| 500 shapes - save / open .blend (ms) | 17.0 / 181.0 | 56.3 / 454.1 |
| 500 shapes - .blend size (bytes) | 16338860 | 1072821 |

## PNG export, 1920 x 1080

### seam.png_export, 100 shapes (strokes, gradients)

|  | blender-4.0.2 | bpy-5.2.2 |
| --- | --- | --- |
| operator, scale 1 (1920 x 1080) | 1026.71 / 2415.53 | 846.15 / 1953.14 |
| render only (no file write), scale 1 | 988.18 / 2015.47 | 822.96 / 1824.01 |
| file size, scale 1 (bytes) | 388886 | 388886 |
| operator, scale 2 (3840 x 2160) | 4587.87 / 6294.49 | 4198.52 / 4229.47 |

## Raster compositing, 2048 x 2048, 4 layers

### median / p95 ms

|  | blender-4.0.2 | bpy-5.2.2 |
| --- | --- | --- |
| numpy kernel `raster.composite` (NORMAL, MULTIPLY 0.8, SCREEN 0.6 + mask, NORMAL 0.9) | 2930.91 / 4396.31 | 4363.20 / 4695.19 |
| `to_uint8_srgb` of the result | 964.40 / 1308.50 | 674.68 / 1146.59 |
| `png.write_png` of the result | 982.03 / 985.83 | 1026.76 / 1106.83 |
| document: `raster_new` 2048 x 2048 (ms, single run) | 153.5 | 380.3 |
| document: `raster_fill` on a selection of 60 % of the width and height | 3699.57 / 5326.73 | 3979.83 / 4808.80 |
| document: `update_composite` (4 layers) | 8599.05 / 9025.09 | 8056.80 / 8447.84 |
| document: `engine.flush()` after a fill of 20 % of the width and height | 8995.60 / 9359.70 | 8386.10 / 9104.88 |
| document: `raster_export_png` composite | 7983.33 / 10263.73 | 8805.83 / 9086.12 |

## SVG, 200 shapes + 3 text objects

### median / p95 ms

|  | blender-4.0.2 | bpy-5.2.2 |
| --- | --- | --- |
| export, text as outlines | 52.94 / 62.56 | 49.44 / 61.06 |
| export, text as text | 49.57 / 120.13 | 43.46 / 72.39 |
| import of the exported file | 223.38 / 258.66 | 249.73 / 262.45 |
| exported file size, outlines (bytes) | 121648 | 121648 |
| shapes after import | 203 | 203 |

## Where one node edit goes in a 500-shape document

Measured with `scripts/bench.py --only diag`: one node edit on a stroked shape with a gradient, then one `engine.flush()`. *depsgraph passes* lists, per `depsgraph_update_post` handler call during that flush, how many datablocks Blender reported as updated (and how many of them were objects). *keys examined* is the number of engine keys looked at by the flush (a key is examined, then skipped when its signature is unchanged). The profile is `cProfile` of one more flush, restricted to add-on and Blender Python frames.

### one flush after one node edit

|  | blender-4.0.2 | bpy-5.2.2 |
| --- | --- | --- |
| depsgraph passes (updates / objects) | 2 / 1; 7 / 2 | 2 / 1; 759 / 752 |
| keys examined per flush | 4 | 1504 |
| keys rebuilt | gradient, stroke | gradient, stroke |

Profile on blender-4.0.2 (top frames by cumulative time):

| function | calls | cumulative ms | self ms |
| --- | --- | --- | --- |
| `extensions/seam_studio/engine/core.py:363 flush` | 1 | 28.5 | 15.0 |
| `extensions/seam_studio/engine/core.py:320 _build_one` | 4 | 7.1 | 0.1 |
| `extensions/seam_studio/engine/core.py:155 _on_depsgraph_update` | 2 | 6.1 | 0.4 |
| `extensions/seam_studio/vector/style.py:659 build` | 1 | 2.9 | 0.0 |
| `extensions/seam_studio/vector/style.py:543 sync_stroke` | 1 | 2.4 | 0.2 |
| `extensions/seam_studio/vector/style.py:428 build` | 1 | 2.0 | 0.0 |
| `extensions/seam_studio/vector/clip.py:318 keys_for_update` | 9 | 1.9 | 0.2 |
| `extensions/seam_studio/vector/canvas.py:734 adoption_hook` | 2 | 1.9 | 0.5 |

Profile on bpy-5.2.2 (top frames by cumulative time):

| function | calls | cumulative ms | self ms |
| --- | --- | --- | --- |
| `extensions/seam_studio/engine/core.py:363 flush` | 1 | 1096.6 | 122.1 |
| `extensions/seam_studio/engine/core.py:320 _build_one` | 1504 | 520.3 | 6.0 |
| `extensions/seam_studio/engine/core.py:155 _on_depsgraph_update` | 2 | 449.7 | 12.0 |
| `extensions/seam_studio/vector/clip.py:318 keys_for_update` | 761 | 341.9 | 1.4 |
| `extensions/seam_studio/vector/clip.py:304 _keys_for_object` | 755 | 340.4 | 2.2 |
| `extensions/seam_studio/vector/style.py:642 stored_signature` | 501 | 256.2 | 3.8 |
| `extensions/seam_studio/vector/clip.py:61 clipped_by` | 503 | 237.8 | 237.8 |
| `extensions/seam_studio/vector/style.py:459 strokes_of` | 251 | 169.6 | 169.6 |

## Observations (computed from the numbers above)

- **blender-4.0.2**: link edits: slowest case is `text` at 20.2 ms p95 (flush after edit); 500-shape document: 25.4 ms median for one node edit, 3401 ms for the first flush after creation; 2048 x 2048 raster: fill 3700 ms, composite rebuild 8599 ms, kernel composite 2931 ms (medians); 1920 x 1080 PNG export: 1027 ms median.
- **bpy-5.2.2**: link edits: slowest case is `text` at 19.3 ms p95 (flush after edit); 500-shape document: 1028.6 ms median for one node edit, 6679 ms for the first flush after creation; 2048 x 2048 raster: fill 3980 ms, composite rebuild 8057 ms, kernel composite 4363 ms (medians); 1920 x 1080 PNG export: 846 ms median.

## Method notes

- Sources are created with the internal shape helpers (fast); everything that is timed as an operator (`seam.make_3d`, `seam.png_export`, `seam.svg_export`, `seam.svg_import`, `seam.raster_*`) goes through `bpy.ops`.
- Edits are Python-side node moves followed by `update_tag()`; this is the same data change a Grab in Edit Mode produces, without the GUI's own redraw cost, which is not measured here.
- Background mode has no timers, so `engine.flush()` stands in for the engine tick (debounce, which is a configurable delay, is not included).
- Not measured: GPU viewport drawing, interactive drawing-tool latency, native sculpt/paint stroke speed (all need the GUI).
