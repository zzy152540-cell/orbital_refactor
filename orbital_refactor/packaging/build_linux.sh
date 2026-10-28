#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
application_dir="$project_root/dist/SatelliteStateEstimation"
executable="$application_dir/SatelliteStateEstimation"
python_executable="${PYTHON_EXE:-python}"

cd "$project_root"
"$python_executable" -m PyInstaller --noconfirm --clean \
  packaging/SatelliteStateEstimation.spec

cp -R configs "$application_dir/configs"
mkdir -p "$application_dir/results"
cp packaging/README_zh.md "$application_dir/README_zh.md"

commit="$(git rev-parse HEAD)"
build_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > "$application_dir/VERSION.txt" <<EOF
product=Satellite State Estimation
version=0.4.0-rc2
git_commit=$commit
platform=Linux-x86_64
built_at_utc=$build_time
EOF

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  smoke_result="$application_dir/results/package_smoke/interface_conformance.json"
  "$executable" --worker-module \
    experiments.run_interface_conformance_acceptance --output "$smoke_result"
  "$python_executable" - "$smoke_result" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if not payload.get("overallPassed"):
    raise SystemExit("Packaged interface verdict is FAIL.")
PY
  QT_QPA_PLATFORM=offscreen "$executable" --gui-smoke
fi

if [[ "${SKIP_ARCHIVE:-0}" != "1" ]]; then
  mkdir -p "$project_root/release"
  tar -czf \
    "$project_root/release/SatelliteStateEstimation-Ubuntu-x86_64-0.4.0-rc2.tar.gz" \
    -C "$project_root/dist" SatelliteStateEstimation
fi

echo "Release package ready: $application_dir"
