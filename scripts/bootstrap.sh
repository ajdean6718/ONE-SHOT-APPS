#!/usr/bin/env bash
# SEAM Studio bootstrap (idempotent). Safe to run any number of times.
#
#   scripts/bootstrap.sh [--rebuild-template] [--no-template] [--quiet]
#
# * checks that a Blender >= 4.0 is available (SEAM_BLENDER overrides the `blender` on PATH),
# * creates the isolated profile  .seam-profile/{config,scripts/addons,scripts/startup/bl_app_templates_user}
#   with symlinks to the add-on and the application template of this checkout,
# * builds application_templates/seam/startup.blend when it is missing (or with --rebuild-template) using
#   scripts/build_template.py in a GUI Blender (xvfb-run when there is no display).
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
PROFILE="${SEAM_PROFILE:-$ROOT/.seam-profile}"
BLENDER="${SEAM_BLENDER:-blender}"
REBUILD=0; BUILD=1; QUIET=0
for a in "$@"; do
  case "$a" in
    --rebuild-template) REBUILD=1 ;;
    --no-template) BUILD=0 ;;
    --quiet) QUIET=1 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done
say() { [ "$QUIET" -eq 1 ] || echo "$@"; }
die() { echo "bootstrap: $*" >&2; exit 1; }

# ---- Blender >= 4.0
if ! command -v "$BLENDER" >/dev/null 2>&1; then
  cat >&2 <<MSG
bootstrap: Blender not found ('$BLENDER').
  Install Blender 4.0 or newer, then re-run this script:
    Ubuntu/Debian:  sudo apt-get install blender
    Any platform:   download the official build from https://www.blender.org/download/ and either put it on PATH
                    or run:  SEAM_BLENDER=/path/to/blender scripts/bootstrap.sh
MSG
  exit 1
fi
VERSION_LINE="$("$BLENDER" --version 2>/dev/null | head -1 || true)"
VERSION="$(printf '%s' "$VERSION_LINE" | sed -n 's/^Blender \([0-9][0-9.]*\).*/\1/p')"
[ -n "$VERSION" ] || die "could not read the Blender version from '$BLENDER --version' (got: '$VERSION_LINE')"
MAJOR="${VERSION%%.*}"
[ "$MAJOR" -ge 4 ] || die "Blender $VERSION is too old; SEAM Studio needs Blender 4.0 or newer"
say "Blender $VERSION ($(command -v "$BLENDER"))"

# ---- add-on layout
ADDON="$ROOT/extensions/seam_studio"
TEMPLATE="$ROOT/application_templates/seam"
[ -f "$ADDON/__init__.py" ] || die "add-on not found at $ADDON"
[ -f "$ADDON/seam_core/__init__.py" ] || die "seam_core is missing in the add-on ($ADDON/seam_core): in a checkout it must be a symlink to src/seam_core"
[ -f "$TEMPLATE/__init__.py" ] || die "application template not found at $TEMPLATE"

# ---- profile
mkdir -p "$PROFILE/config" "$PROFILE/scripts/addons" "$PROFILE/scripts/startup/bl_app_templates_user"
link() {  # link <target> <linkpath>
  if [ -L "$2" ] && [ "$(readlink -f "$2")" = "$(readlink -f "$1")" ]; then return 0; fi
  [ -e "$2" ] && [ ! -L "$2" ] && die "$2 exists and is not a symlink; remove it and re-run"
  ln -sfn "$1" "$2"
}
link "$ADDON" "$PROFILE/scripts/addons/seam_studio"
link "$TEMPLATE" "$PROFILE/scripts/startup/bl_app_templates_user/seam"
say "Profile: $PROFILE"

# ---- startup.blend
STARTUP="$TEMPLATE/startup.blend"
if [ "$BUILD" -eq 1 ] && { [ ! -f "$STARTUP" ] || [ "$REBUILD" -eq 1 ]; }; then
  mkdir -p "$ROOT/artifacts"
  LOG="$ROOT/artifacts/build_template.log"
  RUNNER=()
  if [ -z "${DISPLAY:-}" ]; then
    command -v xvfb-run >/dev/null 2>&1 || die "no display and xvfb-run is missing: install xvfb (apt-get install xvfb) to build startup.blend"
    RUNNER=(xvfb-run -a -s "-screen 0 1920x1080x24")
  fi
  say "Building startup.blend (workspaces Vector, Paint, Model, Split) ..."
  rm -f "$STARTUP"
  if ! BLENDER_USER_CONFIG="$PROFILE/config" BLENDER_USER_SCRIPTS="$PROFILE/scripts" \
       "${RUNNER[@]}" "$BLENDER" --factory-startup --enable-event-simulate --window-geometry 0 0 1920 1080 \
       --python "$ROOT/scripts/build_template.py" -- --out "$STARTUP" >"$LOG" 2>&1; then
    tail -20 "$LOG" >&2
    die "building startup.blend failed (full log: $LOG)"
  fi
  [ -s "$STARTUP" ] || { tail -20 "$LOG" >&2; die "startup.blend was not written (log: $LOG)"; }
  say "Wrote $STARTUP"
fi
[ "$BUILD" -eq 0 ] || [ -f "$STARTUP" ] || die "startup.blend missing"
say "Ready. Start with: $ROOT/scripts/seam-studio"
