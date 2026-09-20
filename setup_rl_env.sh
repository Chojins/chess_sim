#!/usr/bin/env bash
# Recreate the isolated learner runtime without modifying Isaac or the old ACT env.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rl_python="${RL_PYTHON:-/home/jacob/miniconda3/envs/lerobot312/bin/python}"
lerobot_src="${LEROBOT_SRC:-/home/jacob/lerobot/src}"
unset PYTHONPATH PYTHONHOME PYTHONEXE LD_LIBRARY_PATH CUDA_HOME CONDA_PREFIX
export PYTHONNOUSERSITE=1
test -d "$lerobot_src/lerobot" || { echo "Missing LeRobot source: $lerobot_src" >&2; exit 1; }
"$rl_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
"$rl_python" -m venv "$project_dir/.venv-rl"
"$project_dir/.venv-rl/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$project_dir/.venv-rl/bin/python" -m pip install -r "$project_dir/requirements-rl.txt"
"$project_dir/.venv-rl/bin/python" - "$lerobot_src" <<'PY'
from pathlib import Path
import sys
import sysconfig
Path(sysconfig.get_path('purelib'), 'lerobot_source.pth').write_text(str(Path(sys.argv[1]).resolve()) + '\n')
PY
"$project_dir/run_rl_python.sh" -m pip check
