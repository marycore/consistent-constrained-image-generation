# Running ccig-finetuning on RunPod

RunPod-specific commands for this module. The general-purpose sync/tmux tooling lives in the
standalone [`runpod/`](../runpod/README.md) package at the repo root -- this file just shows the
exact sequence for `ccig-finetuning`. See `README.md` in this folder for what the training code
itself does.

All commands below are run from the repo root, using the project name `finetuning` (see
`runpod/projects/finetuning.env`).

## 0) One-time: connection config

```bash
cp runpod/runpod.env.example runpod/runpod.env   # once
# edit runpod/runpod.env with your pod's host/port/key
```

## 1) Sync code + data to the pod

The training data (`data/finetune-dataset/...`) is a separate project mapping from the code, so
sync both the first time:

```bash
./runpod/scripts/sync_to_runpod.sh --project data          # dataset + images (large, first time only)
./runpod/scripts/sync_to_runpod.sh --project finetuning    # this module's code
```

After that first full copy, push just your code edits:

```bash
./runpod/scripts/sync_changed_to_runpod.sh --project finetuning
```

## 2) Bootstrap the remote Python environment

```bash
./runpod/scripts/bootstrap_remote.sh --project finetuning
```

Creates `.venv` under `/workspace/CCIG_Eval/ccig-finetuning` and installs `requirements.txt`.

## 3) Log in to Hugging Face (gated models)

`flux.1-dev` and `sd3.5-large` are gated repos -- accept each model's license on its Hub page
(logged in) once, then log in on the pod non-interactively with a token
(https://huggingface.co/settings/tokens):

```bash
./runpod/scripts/ssh_remote.sh --project finetuning \
  "source .venv/bin/activate && hf auth login --token <your_hf_token>"
```

## 4) Start training in tmux

tmux runs on the pod itself, not on your laptop -- once started, the job keeps running even if
you close your laptop, lose your connection, or the SSH session drops. `start_tmux.sh` creates
the session, redirects output to `logs/<session>.log`, and returns immediately.

**Always export `HF_HOME`/`HUGGINGFACE_HUB_CACHE` pointing at `/workspace`, as below.** RunPod
containers commonly ship a tiny root disk (e.g. 5-20 GB) separate from the large `/workspace`
volume. Hugging Face's download cache defaults to `~/.cache/huggingface` on that small root
disk -- a single gated model like FLUX.1-dev is ~24 GB total, so downloading it without this
export will fill the root disk and fail with a disk-space or "xet" download error partway
through.

```bash
./runpod/scripts/start_tmux.sh --project finetuning flux_dev_stage1 \
  "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache/hub && python -m src.run --model flux.1-dev --config configs/flux.1-dev.yaml --run-name run1"
```

If a previous attempt already filled the root disk, clear the stale cache first:

```bash
./runpod/scripts/ssh_remote.sh --project finetuning "rm -rf /root/.cache/huggingface"
```

Detach any time with `Ctrl+b` then `d` -- this only detaches your terminal, it does not stop the
job.

## 5) Monitor progress

```bash
./runpod/scripts/list_tmux.sh --project finetuning                          # is it still running?
./runpod/scripts/tail_log.sh --project finetuning flux_dev_stage1           # live log stream
./runpod/scripts/attach_tmux.sh --project finetuning flux_dev_stage1        # reattach to the session
```

### Stopping a run

```bash
./runpod/scripts/stop_tmux.sh --project finetuning flux_dev_stage1
```

Kills the tmux session immediately. **This does not save a checkpoint** unless the run had
already reached `max_steps` on its own -- `train()` only calls `save_pretrained` once, at the
very end of the training loop (see `src/models/_diffusers_common.py`), so anything trained since
the last save is lost when you stop it mid-run.

## 6) Pull the checkpoint back once training finishes

```bash
./runpod/scripts/sync_folder_from_runpod.sh --project finetuning --folder outputs
```

Merges `outputs/flux.1-dev/run1/` (adapters + `train_config.json` + `train_log.jsonl`) into your
local `ccig-finetuning/outputs/`. Add `--delete` if you want an exact mirror instead of a merge.

## Other models

Same steps, just change `--model`/`--config`/tmux session name:

```bash
./runpod/scripts/start_tmux.sh --project finetuning sd35_large_stage1 \
  "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache/hub && python -m src.run --model sd3.5-large --config configs/sd3.5-large.yaml --run-name run1"

./runpod/scripts/start_tmux.sh --project finetuning qwen_image_stage1 \
  "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache/hub && python -m src.run --model qwen-image --config configs/qwen-image.yaml --run-name run1"
```

## Troubleshooting

See the [`runpod/README.md`](../runpod/README.md#troubleshooting) troubleshooting section for
connection issues (refused connections, host-key errors, missing `.venv`, etc).
