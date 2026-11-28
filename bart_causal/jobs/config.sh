# Settings for the BART causal-link jobs. Edit these once; every job reads them.
ACCOUNT=""              # your project / allocation code (sbatch --account); leave empty if your cluster needs none
CPU_PARTITION=""        # partition for CPU jobs (empty = cluster default)
GPU_PARTITION=""        # partition with GPUs (empty = cluster default)
GPU_GRES="gpu:1"        # how your cluster asks for one GPU (e.g. gpu:1, gpu:a100:1)
MODULES=""              # modules to load first, e.g. "python/3.11.3 cuda/12.2.0" (run `module avail python cuda`)

RUN="runs/r1"                            # output folder (relative to bart_causal/)
MODELS="$HOME/models"                    # where setup_env.sh downloads the models
BART="$MODELS/bart-large-mnli"           # the model to fine-tune
LLM="$MODELS/Qwen2.5-7B-Instruct"        # the judge for labelling; any local chat model works (your Qwen path)
N_LABELS=6000                            # pairs the judge labels (more = better model, longer job)
EPOCHS=3
