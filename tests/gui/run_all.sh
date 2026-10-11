#!/usr/bin/env bash
# Run every GUI scenario (tests/gui/gui_*.py) in its own Xvfb + GUI Blender with a fresh isolated profile and the
# SEAM application template. Exit code: 0 all passed (skips allowed), 1 any failure.
#
#   tests/gui/run_all.sh [scenario-name ...]      e.g. tests/gui/run_all.sh gui_launch
#
# Screenshots + result.json land in artifacts/gui/<scenario>/. SEAM_BLENDER overrides the Blender binary.
set -uo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
cd "$ROOT"
BLENDER="${SEAM_BLENDER:-$(command -v blender || true)}"
[ -n "$BLENDER" ] || { echo "run_all: blender not found (set SEAM_BLENDER)"; exit 2; }
command -v xvfb-run >/dev/null 2>&1 || { echo "run_all: xvfb-run not found (apt-get install xvfb)"; exit 2; }
command -v import >/dev/null 2>&1 || { echo "run_all: ImageMagick 'import' not found (apt-get install imagemagick)"; exit 2; }
TIMEOUT="${SEAM_GUI_TIMEOUT:-240}"
GEOM="${SEAM_GUI_GEOMETRY:-1600x900}"

# the template needs startup.blend (built from the current add-on)
if [ ! -f application_templates/seam/startup.blend ]; then
  scripts/bootstrap.sh --quiet || { echo "run_all: bootstrap failed"; exit 2; }
fi

if [ "$#" -gt 0 ]; then SCENARIOS=("$@"); else
  mapfile -t SCENARIOS < <(ls tests/gui/gui_*.py | xargs -n1 basename | sed 's/\.py$//' | sort)
fi

mkdir -p artifacts/gui
RC=0; PASSED=0; FAILED=0; SKIPPED=0; SUMMARY=()
for name in "${SCENARIOS[@]}"; do
  script="tests/gui/${name}.py"
  [ -f "$script" ] || { echo "run_all: no such scenario: $name"; RC=1; continue; }
  PROFILE="$(mktemp -d -t seam-gui-XXXXXX)"
  mkdir -p "$PROFILE/config" "$PROFILE/scripts/addons" "$PROFILE/scripts/startup/bl_app_templates_user"
  ln -s "$ROOT/extensions/seam_studio" "$PROFILE/scripts/addons/seam_studio"
  ln -s "$ROOT/application_templates/seam" "$PROFILE/scripts/startup/bl_app_templates_user/seam"
  OUT="$ROOT/artifacts/gui/$name"
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "== $name"
  BLENDER_USER_CONFIG="$PROFILE/config" BLENDER_USER_SCRIPTS="$PROFILE/scripts" \
  SEAM_GUI_OUT="$OUT" SEAM_GUI_TIMEOUT="$TIMEOUT" \
  timeout $((TIMEOUT + 60)) xvfb-run -a -s "-screen 0 ${GEOM}x24" \
    "$BLENDER" --app-template seam --enable-event-simulate --window-geometry 0 0 "${GEOM%x*}" "${GEOM#*x}" \
    --python "$ROOT/$script" >"$OUT/blender.log" 2>&1
  code=$?
  rm -rf "$PROFILE"
  case "$code" in
    0)  echo "   PASS"; PASSED=$((PASSED+1)); SUMMARY+=("PASS  $name") ;;
    77) reason="$(grep -h 'SKIP:' "$OUT/blender.log" | tail -1)"; echo "   SKIP ${reason}"; SKIPPED=$((SKIPPED+1)); SUMMARY+=("SKIP  $name  ${reason}") ;;
    *)  echo "   FAIL (exit $code) see $OUT/blender.log"; grep -h '\[gui:.*FAIL' "$OUT/blender.log" | head -10
        FAILED=$((FAILED+1)); RC=1; SUMMARY+=("FAIL  $name (exit $code)") ;;
  esac
done
echo
printf '%s\n' "${SUMMARY[@]}"
echo "gui: $PASSED passed, $FAILED failed, $SKIPPED skipped"
exit $RC
