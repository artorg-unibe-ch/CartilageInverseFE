#!/bin/bash
#SBATCH --job-name=VE_native_test_newData2
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/homefs/kx24c057/Cartilage/logs3/%x_%j.out
#SBATCH --error=/storage/homefs/kx24c057/Cartilage/logs3/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=keke.xia@students.unibe.ch

set -u

SCRIPT_PATH="/storage/homefs/kx24c057/Cartilage/scripts4/Submit_VE_native_autosumit.sh"
RESULTS_CSV="/storage/homefs/kx24c057/Cartilage/NewExperimentData_for_Optimization/Native_cartilage_csv_for45dModel/VE_healthy.csv"
EXPECTED_N=25
MAX_CHAIN=20

CHAIN_COUNT=${CHAIN_COUNT:-0}

echo "Job ID: $SLURM_JOB_ID"
echo "Chain count: $CHAIN_COUNT"

# 如果已经完成，就直接退出，不再提交
if [[ -f "$RESULTS_CSV" ]]; then
    DONE_N=$(($(wc -l < "$RESULTS_CSV") - 1))
    if [[ "$DONE_N" -ge "$EXPECTED_N" ]]; then
        echo "All samples completed: $DONE_N / $EXPECTED_N"
        exit 0
    fi
fi

# 防止无限循环提交
if [[ "$CHAIN_COUNT" -lt "$MAX_CHAIN" ]]; then
    NEXT_CHAIN=$((CHAIN_COUNT + 1))
    NEXT_JOB_ID=$(sbatch --parsable \
        --dependency=afterany:${SLURM_JOB_ID} \
        --export=ALL,CHAIN_COUNT=${NEXT_CHAIN} \
        "$SCRIPT_PATH")

    echo "Submitted next chained job: $NEXT_JOB_ID"
else
    echo "Reached MAX_CHAIN=$MAX_CHAIN, will not submit more jobs."
    NEXT_JOB_ID=""
fi

# 如果本 job 正常完成全部样本，退出时取消已经预约的下一个 job
cleanup() {
    if [[ -n "${NEXT_JOB_ID:-}" && -f "$RESULTS_CSV" ]]; then
        DONE_N=$(($(wc -l < "$RESULTS_CSV") - 1))
        if [[ "$DONE_N" -ge "$EXPECTED_N" ]]; then
            echo "Completed all samples. Cancel next job $NEXT_JOB_ID"
            scancel "$NEXT_JOB_ID"
        fi
    fi
}
trap cleanup EXIT

module load FFTW/3.3.10-GCC-12.3.0
module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AC_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python /storage/homefs/kx24c057/Cartilage/scripts4/Native_VE_singleStart_test.py