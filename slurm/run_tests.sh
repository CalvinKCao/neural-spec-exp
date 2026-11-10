#!/bin/bash
# CPU-only pytest run on Narval (no GPU, no torch: networks come from artifacts/*.npz).
#
# USAGE (login node, from $SCRATCH/verify-exp/nap_lp):
#   sbatch slurm/run_tests.sh
#SBATCH --job-name=nap-tests
#SBATCH --account=def-boyuwang
#SBATCH --time=02:59:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p results/logs
LOG="results/logs/$(date +%m-%d)-${SLURM_JOB_ID: -3}-pytest-narval.log"
exec >>"$LOG" 2>&1

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME, started $(date)"
module purge || true
module load StdEnv/2023
module load python/3.12

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index numpy scipy pytest pandas matplotlib -q
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NAP_MILP_TL=60 NAP_TEST_PROCS=8

python -m pytest tests
echo "pytest exit $? at $(date)"
cp results/test_report.json "results/test_report_narval.json" 2>/dev/null || true
