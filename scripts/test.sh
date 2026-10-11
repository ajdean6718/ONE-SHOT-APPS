#!/usr/bin/env bash
# SEAM Studio test runner. Usage: scripts/test.sh [unit|int40|int52|acc40|acc52|gui|all] [extra pytest args...]
# Logs: artifacts/test-logs/<suite>.log ; exit code = first failing suite's code.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SUITE="${1:-all}"; shift || true
LOGDIR="$ROOT/artifacts/test-logs"; mkdir -p "$LOGDIR"
VENV52="${SEAM_BPY52_PYTHON:-/opt/seam-cache/venv-bpy52/bin/python}"
BLENDER40="${SEAM_BLENDER:-$(command -v blender || true)}"
PY312="${SEAM_PY312:-/usr/bin/python3.12}"

isolated_profile() {
  local d; d="$(mktemp -d -t seam-test-XXXXXX)"
  mkdir -p "$d/scripts/addons" "$d/config"
  ln -s "$ROOT/extensions/seam_studio" "$d/scripts/addons/seam_studio"
  export BLENDER_USER_SCRIPTS="$d/scripts" BLENDER_USER_CONFIG="$d/config"
}

run_unit() {
  local rc=0
  echo "== unit (bpy 5.2 venv python $("$VENV52" -c 'import sys;print(sys.version.split()[0])'))"
  PYTHONPATH="$ROOT/src" "$VENV52" -m pytest -q -p no:cacheprovider tests/unit "$@" 2>&1 | tee "$LOGDIR/unit-py313.log"; rc=${PIPESTATUS[0]}
  if [ -x "$PY312" ]; then
    echo "== unit (system python 3.12, Blender 4.0 runtime libs)"
    PYTHONPATH="$ROOT/src" "$PY312" -m pytest -q -p no:cacheprovider tests/unit "$@" 2>&1 | tee "$LOGDIR/unit-py312.log"; local rc2=${PIPESTATUS[0]}
    [ $rc -eq 0 ] && rc=$rc2
  fi
  return $rc
}

run_int52() {
  isolated_profile
  echo "== integration (Blender 5.2.2 LTS bpy module)"
  "$VENV52" -m pytest -q -p no:cacheprovider tests/integration "$@" 2>&1 | tee "$LOGDIR/int52.log"; return ${PIPESTATUS[0]}
}

run_int40() {
  isolated_profile
  echo "== integration (Blender $("$BLENDER40" --version 2>/dev/null | head -1 | awk '{print $2}') binary, background)"
  "$BLENDER40" -b --factory-startup --python-exit-code 1 --python "$ROOT/scripts/blender_pytest.py" -- -q tests/integration "$@" 2>&1 | tee "$LOGDIR/int40.log"; return ${PIPESTATUS[0]}
}

run_acc52() {
  isolated_profile
  echo "== acceptance (Blender 5.2.2 LTS bpy module)"
  "$VENV52" -m pytest -q -p no:cacheprovider tests/acceptance "$@" 2>&1 | tee "$LOGDIR/acc52.log"; return ${PIPESTATUS[0]}
}

run_acc40() {
  isolated_profile
  echo "== acceptance (Blender 4.0.2 binary, background)"
  "$BLENDER40" -b --factory-startup --python-exit-code 1 --python "$ROOT/scripts/blender_pytest.py" -- -q tests/acceptance "$@" 2>&1 | tee "$LOGDIR/acc40.log"; return ${PIPESTATUS[0]}
}

run_gui() {
  if [ -x "$ROOT/tests/gui/run_all.sh" ]; then
    "$ROOT/tests/gui/run_all.sh" "$@" 2>&1 | tee "$LOGDIR/gui.log"; return ${PIPESTATUS[0]}
  fi
  echo "GUI suite not present (tests/gui/run_all.sh)"; return 3
}

case "$SUITE" in
  unit) run_unit "$@" ;;
  int52) run_int52 "$@" ;;
  int40) run_int40 "$@" ;;
  acc52) run_acc52 "$@" ;;
  acc40) run_acc40 "$@" ;;
  gui) run_gui "$@" ;;
  all)
    rc=0
    for s in run_unit run_int52 run_int40 run_acc52 run_acc40 run_gui; do $s "$@"; r=$?; [ $rc -eq 0 ] && rc=$r; done
    exit $rc ;;
  *) echo "unknown suite $SUITE"; exit 2 ;;
esac
