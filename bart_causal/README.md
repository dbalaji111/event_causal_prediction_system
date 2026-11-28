# BART causal-link training (supercomputer)

Trains a BART model that reads two gold events, A's **effect** text and a later event B's **cause** text, and scores whether A
plausibly led to B. The causal-chain view (`redflag_market_kit/demo/export_causal_chains.py`) currently links events by TF-IDF
word overlap; this replaces that score with the trained model's.

Input: `scripts/causal_gold_articles_with_topics_bert.csv` (already in the repo). Nothing else needs copying to the cluster.

## Stages

| # | stage | what it does | runs on | writes (in `runs/r1/`) |
|---|-------|--------------|---------|------------------------|
| 1 | `pairs` | for each event A, the later events B within 120 days that are closest by TF-IDF (top 20), plus 5 random ones | CPU, minutes | `pairs.csv` |
| 2 | `label` | a local Qwen judges 6000 sampled pairs: "did A plausibly contribute to B?"; P(Yes) from its next-token logits | GPU | `labels.csv` |
| 3 | `train` | fine-tunes `facebook/bart-large-mnli` (entailment = link). Time split: the latest 20% of source dates are held out | GPU, ~1 h | `model/`, `train_metrics.json` |
| 4 | `score` | scores every candidate pair: the fine-tuned model and the untrained MNLI model as a baseline | GPU | `bart_link_scores.csv`, `bart_link_scores_zeroshot.csv` |
| 5 | `eval` | compares BART, zero-shot BART, TF-IDF and embedding similarity on the held-out labels (AUC, AP, F1, precision@k) | CPU | `eval.json` |

## Running it

```bash
git clone -b claude/bart-causal-training https://github.com/dbalaji111/event_causal_prediction_system.git
cd event_causal_prediction_system/bart_causal

nano jobs/config.sh          # 1. your account, partitions, GPU request, modules, model paths
bash jobs/setup_env.sh       # 2. on the LOGIN node (needs internet): .venv + model downloads
bash jobs/run_all.sh         # 3. submits the 5 jobs; each waits for the previous one (afterok)

squeue -u $USER              # progress
tail -f logs/bart-train-*.out
cat runs/r1/eval.json        # results
```

Re-run from one stage onwards (e.g. after changing training settings): `bash jobs/run_all.sh train`.

Without Slurm, or for a quick check on one GPU node, run the stages directly:

```bash
source .venv/bin/activate
python bart_causal_pipeline.py pairs --out runs/test --limit 300
python bart_causal_pipeline.py label --out runs/test --labeler heuristic --n 500     # no LLM needed
python bart_causal_pipeline.py train --out runs/test --model ~/models/bart-large-mnli --epochs 1
python bart_causal_pipeline.py score --out runs/test
python bart_causal_pipeline.py eval  --out runs/test
```

## What to set in `jobs/config.sh`

- `ACCOUNT`, `CPU_PARTITION`, `GPU_PARTITION`: from your cluster's docs or `sinfo` (leave empty if not needed).
- `GPU_GRES`: usually `gpu:1`; some clusters want the type, e.g. `gpu:a100:1`.
- `MODULES`: e.g. `python/3.11.3 cuda/12.2.0` (`module avail python cuda`).
- `LLM`: the Qwen you already have. A 7B instruct model fits one 40 GB GPU in bf16; a 14B+ needs 80 GB.
- Compute nodes usually have no internet, so every job runs with `HF_HUB_OFFLINE=1` and loads the models from `MODELS`.

## Using the result in the chain view

Copy `runs/r1/bart_link_scores.csv` back to your machine, then:

```bash
python redflag_market_kit/demo/export_causal_chains.py --bart-scores bart_causal/runs/r1/bart_link_scores.csv
```

It keeps up to 4 links per event (gap 1-120 days) with a score of at least 0.5 (`MIN_BART_SCORE`), and marks the snapshot's
`link_method` as `bart`. Without `--bart-scores` it still uses TF-IDF.

## Read the results carefully

- The labels come from an LLM judge, so they are a teacher signal, not ground truth. `eval.json` says how well BART reproduces
  the judge on later, unseen events; it does not prove the links are causal in the market. Hand-check about 100 pairs before
  trusting it, and before quoting it to anyone.
- Compare against the baselines in `eval.json`: if fine-tuned BART does not beat TF-IDF and zero-shot BART, it is not worth using.
- `label` warns if fewer than 2% or more than 98% of the pairs come out linked; `train` stops if the labels are all one class.
  Either means the judge prompt or `--threshold` needs changing.
- `runs/` and `logs/` are git-ignored: the outputs contain article text, so don't commit them.

Tested here end to end with small random stand-in models only (this environment cannot download from Hugging Face); the first
real run on the cluster is the actual test.
