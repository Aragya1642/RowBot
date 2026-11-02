#!/usr/bin/env bash
# RowBOT physics smoke test (headless xdyn, no Gazebo/Unity).
# Run INSIDE the LOTUSim dev shell:   cd ~/LOTUSim && nix develop
# then:                               bash ~/rowbot_smoke/run_smoke.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LOTUSIM_REPO="${LOTUSIM_REPO:-$HOME/LOTUSim}"
MESH="$LOTUSIM_REPO/assets/models/dtmb_hull/meshes/dtmb.stl"
DT="${DT:-0.05}"        # s
TEND="${TEND:-300}"     # s

command -v xdyn >/dev/null || { echo "xdyn not on PATH. Run this inside: cd $LOTUSIM_REPO && nix develop"; exit 1; }
[ -f "$MESH" ] || { echo "Mesh not found at $MESH. Clone LOTUSim to ~/LOTUSim or set LOTUSIM_REPO."; exit 1; }

OUT="$HERE/out"
mkdir -p "$OUT"
cd "$OUT"

make_yml () {  # $1 = snippet, $2 = csv name, $3 = yml name
  awk -v snip="$HERE/$1" '
    /^@WAVES@$/ { while ((getline line < snip) > 0) print line; next }
    { print }' "$HERE/dtmb_smoke.yml.in" \
  | sed -e "s|@MESH@|$MESH|" -e "s|@CSV@|$2|" > "$3"
}

make_yml waves_regular.snippet waves.csv dtmb_waves.yml
make_yml waves_flat.snippet    calm.csv  dtmb_calm.yml

for case in calm waves; do
  echo "== Running $case (dt=$DT s, tend=$TEND s) =="
  start=$(date +%s)
  xdyn "dtmb_$case.yml" --dt "$DT" --tend "$TEND" > "xdyn_$case.log" 2>&1 \
    || { echo "xdyn failed for $case, see $OUT/xdyn_$case.log"; tail -20 "xdyn_$case.log"; exit 1; }
  wall=$(( $(date +%s) - start )); [ "$wall" -lt 1 ] && wall=1
  echo "   done in ${wall} s wall clock, about $(( ${TEND%.*} / wall ))x faster than real time"
done

python3 "$HERE/analyze_smoke.py" calm.csv waves.csv
