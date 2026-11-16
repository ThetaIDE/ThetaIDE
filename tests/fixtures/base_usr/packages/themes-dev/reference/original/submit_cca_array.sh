#!/bin/bash
# =============================================================================
# SLURM Job Array — CCA (Contrastive-Critical Apprenticeship Learning)
# 10 repeats, one node per repeat, all run simultaneously.
#
# Architecture: Phase 0 (load pre-computed InferGP w_t_gp weights) →
#               Phase 1 (MT-TICC clustering on all training data) →
#               Phase 2 (Dual-arm kNN criticality: Q10/Q11 hi + Q00/Q01 lo) →
#               Phase 3 [EM-EDM on Q10+Q00 only, arm-routed joint weights] →
#               Phase 4 [BT-IRL: D+ = Q10∪Q00, D- = Q11∪Q01, arm-routed w±] →
#               REG-MT-TICC → re-cluster
#
# Submit examples:
#   sbatch --export=ALL,EHR_ENV=CCHS,DATA_MODE=all,RUN_TAG=v1   submit_cca_array.sh
#   sbatch --export=ALL,EHR_ENV=Mayo,DATA_MODE=all,RUN_TAG=v1   submit_cca_array.sh
#   sbatch --export=ALL,EHR_ENV=MIMIC,DATA_MODE=all,RUN_TAG=v1  submit_cca_array.sh
#
# Ablations — disable individual CCA components:
#   sbatch --export=ALL,EHR_ENV=CCHS,DATA_MODE=all,GP_CREDIT=0,RUN_TAG=nogp          submit_cca_array.sh
#   sbatch --export=ALL,EHR_ENV=CCHS,DATA_MODE=all,USE_CRITICALITY=0,RUN_TAG=nocrit  submit_cca_array.sh
#   sbatch --export=ALL,EHR_ENV=CCHS,DATA_MODE=all,LAMBDA_CONTRAST=0,RUN_TAG=noclc   submit_cca_array.sh
#
# kNN criticality sweep:
#   sbatch --export=ALL,EHR_ENV=CCHS,DATA_MODE=all,KNN_NEIGHBORS=30,RUN_TAG=knn30    submit_cca_array.sh
#   sbatch --export=ALL,EHR_ENV=CCHS,DATA_MODE=all,KNN_NEIGHBORS=100,RUN_TAG=knn100  submit_cca_array.sh
#
# Monitor:  squeue -u $USER
# Logs:     logs/cca_<JobID>_<TaskID>.out
#
# After all tasks finish, aggregate:
#   MODEL=CCA EHR_ENV=CCHS USE_HALIDE_W=1 DATA_MODE=all RUN_TAG=v1 \
#     python THEMES/THEMES_src/aggregate_results.py
#
# Key env vars:
#   EHR_ENV            CCHS / Mayo / MIMIC              (default CCHS)
#   DATA_MODE          all / ei / e                     (default all)
#   USE_HALIDE_W       use HALIDE traj weights          (default 1)
#   GP_CREDIT          load InferGP w_t_gp column       (default 1)
#   USE_CRITICALITY    enable dual-arm criticality       (default 1)
#   KNN_NEIGHBORS      k for kNN criticality index       (default 50)
#   MIN_CRIT_SUPPORT   min kNN support for C(s_t)        (default 3)
#   LAMBDA_CONTRAST    gate: >0 enables BT-IRL, =0 falls back to standard MLIRL (default 0.5)
#   CRIT_IN_PHASE3     1 (default)=criticality in both EM-EDM and BT-IRL; 0=Phase 4 only
#   CRIT_WEIGHT_CAP    cap after mean=1 normalisation (default 2.0; was 3.0 — 3.0 creates bimodal weights)
#   NEG_QUADRANT       anti-expert quadrant for hi-arm: 11=shock→shock (default), 01=nonshock→shock
#   NRCL               EM-EDM cluster count              (default 3)
#   GP_WEIGHT_COL      GP weight column: w_t_gp (default) or w_t_gp_norm
#   MAX_NONEXPERT_RATIO  cap each non-expert quadrant at floor(ratio/3 × Q10)
#                        default 3 → 1:1:1:1 balance (Q10 is smallest in all datasets)
#   RUN_TAG            version tag for result filenames
# =============================================================================

#SBATCH -J CCA                      # job name
#SBATCH -N 1                        # 1 node per array task
#SBATCH -n 4                        # 4 CPU cores per node
#SBATCH -p rtx4060ti16g             # GPU partition
#SBATCH --time=24:00:00             # wall time (10 outer iters with BT pairwise IRL)
#SBATCH --array=0-9                 # 10 repeats
#SBATCH --output=logs/cca_%A_%a.out
#SBATCH --error=logs/cca_%A_%a.err
#SBATCH --exclude=c2,c23,c24,c51,c65,c66,c67,c69,c73

# --------------------------------------------------------------------------
# Resolve env-var config with defaults
# --------------------------------------------------------------------------
EHR_ENV=${EHR_ENV:-CCHS}
DATA_MODE=${DATA_MODE:-all}
USE_HALIDE_W=${USE_HALIDE_W:-1}
GP_CREDIT=${GP_CREDIT:-1}
USE_CRITICALITY=${USE_CRITICALITY:-1}
CRIT_IN_PHASE3=${CRIT_IN_PHASE3:-1}     # 0 = criticality in Phase 4 (BT-IRL) only, not Phase 3 (EM-EDM)
CRIT_WEIGHT_CAP=${CRIT_WEIGHT_CAP:-2.0} # asymptote for tanh compression (or hard-clip ceiling if CRIT_SOFT_CAP=0)
CRIT_SOFT_CAP=${CRIT_SOFT_CAP:-1}       # 1 = tanh(x/cap)*cap (continuous); 0 = hard clip (legacy, bimodal)
CRIT_P4_MODE=${CRIT_P4_MODE:-jsd}       # jsd (default) | p_action (legacy kNN P±) | none (GP only in Phase 4)
KNN_NEIGHBORS=${KNN_NEIGHBORS:-50}
MIN_CRIT_SUPPORT=${MIN_CRIT_SUPPORT:-3}
LAMBDA_CONTRAST=${LAMBDA_CONTRAST:-0.5}
NEG_QUADRANT=${NEG_QUADRANT:-11}
NRCL=${NRCL:-3}
HALIDE_VARIANT=${HALIDE_VARIANT:-}
GP_WEIGHT_COL=${GP_WEIGHT_COL:-w_t_gp_norm}
# MAX_NONEXPERT_RATIO=3 caps each of Q11/Q00/Q01 at floor(3/3 × Q10) = Q10 count.
# Q10 is the smallest quadrant in all datasets (CCHS:119, Mayo:75, MIMIC:116), so this
# produces 1:1:1:1 balance across all quadrants — critical for CCA because:
#   (a) EM-EDM Phase 3 uses Q10+Q00: without cap, Q00 is 10-24× Q10, dominating clusters.
#   (b) Hi-arm kNN (Q10 vs Q11): Q11 is 5-20× Q10, making Q10 the sparse positive pool.
#   (c) Lo-arm kNN (Q00 vs Q01): Q01 is small but Q00 is huge — repeat neighbor aliasing.
MAX_NONEXPERT_RATIO=${MAX_NONEXPERT_RATIO:-3}
RUN_TAG=${RUN_TAG:-v1}

# --------------------------------------------------------------------------
# Ensure log directory exists (must happen before SLURM writes .out/.err)
# --------------------------------------------------------------------------
mkdir -p logs

# --------------------------------------------------------------------------
# Pre-flight: verify required model config exists
# --------------------------------------------------------------------------
BASE="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="${BASE}/models/EHRs/${EHR_ENV}/EM-EDM_models/para_config.npy"
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: para_config.npy not found at ${CONFIG_PATH}" >&2
    echo "       Run the base EM-EDM training for ${EHR_ENV} before CCA." >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Pre-create directories that the Python script expects to exist at startup.
# (Results and model subdirs are created by the script itself via makedirs;
#  these are only the top-level anchors that must pre-exist.)
# --------------------------------------------------------------------------
mkdir -p "${BASE}/results/EHRs/${EHR_ENV}/CCA"
mkdir -p "${BASE}/models/EHRs/${EHR_ENV}/EM-EDM_models"
mkdir -p "${BASE}/volume/EHRs/${EHR_ENV}/MTTICC"
mkdir -p "${BASE}/volume/EHRs/${EHR_ENV}/CCA_${DATA_MODE}"
mkdir -p "${BASE}/logs"

# --------------------------------------------------------------------------
echo "=============================="
echo "Job ID             : $SLURM_JOB_ID"
echo "Array Task         : $SLURM_ARRAY_TASK_ID  (= rp_idx)"
echo "Node               : $(hostname)"
echo "Start time         : $(date)"
echo "EHR_ENV            : $EHR_ENV"
echo "DATA_MODE          : $DATA_MODE"
echo "USE_HALIDE_W       : $USE_HALIDE_W"
echo "GP_CREDIT          : $GP_CREDIT"
echo "USE_CRITICALITY    : $USE_CRITICALITY"
echo "CRIT_IN_PHASE3     : $CRIT_IN_PHASE3"
echo "CRIT_WEIGHT_CAP    : $CRIT_WEIGHT_CAP"
echo "CRIT_SOFT_CAP      : $CRIT_SOFT_CAP"
echo "CRIT_P4_MODE       : $CRIT_P4_MODE"
echo "KNN_NEIGHBORS      : $KNN_NEIGHBORS"
echo "MIN_CRIT_SUPPORT   : $MIN_CRIT_SUPPORT"
echo "LAMBDA_CONTRAST    : $LAMBDA_CONTRAST"
echo "NEG_QUADRANT       : $NEG_QUADRANT"
echo "NRCL               : $NRCL"
echo "HALIDE_VARIANT     : ${HALIDE_VARIANT:-default}"
echo "GP_WEIGHT_COL      : $GP_WEIGHT_COL"
echo "MAX_NONEXPERT_RATIO: $MAX_NONEXPERT_RATIO"
echo "RUN_TAG            : $RUN_TAG"
echo "CONFIG_PATH        : $CONFIG_PATH"
echo "=============================="

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate themes_gpu
else
    echo "Warning: conda not found; skipping conda activate" >&2
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    && echo "GPU detected." || echo "WARNING: nvidia-smi failed — no GPU visible."

scontrol update JobId="${SLURM_ARRAY_JOB_ID}" JobName="CCA_${EHR_ENV}" 2>/dev/null || true

cd "${BASE}"

TICC_NUM_PROC=4 \
TICC_TRAIN=1 \
REG_TICC_TRAIN=1 \
MAX_CLUSTER_WORKERS=3 \
TORCH_COMPILE=0 \
EHR_ENV=${EHR_ENV} \
DATA_MODE=${DATA_MODE} \
USE_HALIDE_W=${USE_HALIDE_W} \
GP_CREDIT=${GP_CREDIT} \
USE_CRITICALITY=${USE_CRITICALITY} \
CRIT_IN_PHASE3=${CRIT_IN_PHASE3} \
CRIT_WEIGHT_CAP=${CRIT_WEIGHT_CAP} \
CRIT_SOFT_CAP=${CRIT_SOFT_CAP} \
CRIT_P4_MODE=${CRIT_P4_MODE} \
KNN_NEIGHBORS=${KNN_NEIGHBORS} \
MIN_CRIT_SUPPORT=${MIN_CRIT_SUPPORT} \
LAMBDA_CONTRAST=${LAMBDA_CONTRAST} \
NEG_QUADRANT=${NEG_QUADRANT} \
NRCL=${NRCL} \
HALIDE_VARIANT=${HALIDE_VARIANT} \
GP_WEIGHT_COL=${GP_WEIGHT_COL} \
MAX_NONEXPERT_RATIO=${MAX_NONEXPERT_RATIO} \
RUN_TAG=${RUN_TAG} \
python -u cca_learn.py

echo "=============================="
echo "End time : $(date)"
echo "=============================="
