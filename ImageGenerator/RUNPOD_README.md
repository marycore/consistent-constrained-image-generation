# RunPod Persistent-Volume Runbook

Use this when you create/delete pods often and keep code/data on a mounted volume.

## 0) Edit connection config

File: `scripts/runpod/runpod.env`

```env
RUNPOD_HOST=<pod_ip_or_dns>
RUNPOD_USER=root
RUNPOD_PORT=<ssh_port>
RUNPOD_SSH_KEY=~/.ssh/id_ed25519
RUNPOD_REMOTE_DIR=/workspace/CCIG_Eval/ImageGenerator
```

Notes:
- Update `RUNPOD_HOST` and `RUNPOD_PORT` every time you create a new pod.
- Keep `RUNPOD_REMOTE_DIR` on the mounted volume path (typically under `/workspace`).

## 0.5) Connect from local terminal (always)

From local `ImageGenerator` directory, use one of these:

```bash
# Preferred: use wrapper (reads scripts/runpod/runpod.env)
./scripts/runpod/ssh_remote.sh "hostname && pwd"

# Direct SSH (same values as runpod.env)
ssh "${RUNPOD_USER}@${RUNPOD_HOST}" -p "${RUNPOD_PORT}" -i "${RUNPOD_SSH_KEY/#\~/$HOME}"
```

Quick sanity checks after connecting:

```bash
./scripts/runpod/ssh_remote.sh "hostname && pwd && ls -lah"
./scripts/runpod/ssh_remote.sh "df -h | sed -n '1p;/workspace/p'"
```

## 1) One-time checks on a new pod

Run from local `ImageGenerator` directory.

```bash
# Accept host key and verify mount/GPU
./scripts/runpod/ssh_remote.sh "hostname && pwd && df -h && nvidia-smi"

# Install remote basics if needed
./scripts/runpod/ssh_remote.sh "apt-get update && apt-get install -y rsync tmux"
```

## 2) First-time copy to the pod/volume

Use full mirror when initializing a new pod or when you need exact parity.

```bash
./scripts/runpod/sync_to_runpod.sh
```

Then bootstrap Python environment:

```bash
./scripts/runpod/bootstrap_remote.sh
```

## 3) Daily workflow (sync only changes)

For normal iteration, use incremental sync:

```bash
./scripts/runpod/sync_changed_to_runpod.sh
```

Important:
- `sync_changed_to_runpod.sh` is git-based and best for source-code edits.
- If you changed many generated/ignored files and need exact parity, use `sync_to_runpod.sh`.

## 4) Start training in tmux

Example: PixArt stage 1

```bash
./scripts/runpod/start_tmux.sh pixart_stage1 "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ImageGenerator/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ImageGenerator/.hf_cache/hub && python -m src.common.cli finetune --model pixart_sigma --data configs/finetune_dataset/dataset.json --images_root configs/finetune_dataset/images --caption_key general_text --max_sequence_length 256 --out ckpts/pixart_sigma_lora_stage1 --max_steps 200 --lr 1e-4 --batch_size 1 --grad_accum 4 --seed 42"
```

Example: PixArt stage 2 from stage 1

```bash
./scripts/runpod/start_tmux.sh pixart_stage2 "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ImageGenerator/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ImageGenerator/.hf_cache/hub && python -m src.common.cli finetune --model pixart_sigma --data configs/finetune_dataset/dataset.json --images_root configs/finetune_dataset/images --caption_key pred --max_sequence_length 256 --out ckpts/pixart_sigma_lora_stage2 --init_ckpt ckpts/pixart_sigma_lora_stage1 --max_steps 200 --lr 5e-5 --batch_size 1 --grad_accum 4 --seed 42"
```

Example: SD3.5 stage 1 (experimental; high VRAM recommended)

```bash
./scripts/runpod/start_tmux.sh sd3_5_stage1 "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ImageGenerator/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ImageGenerator/.hf_cache/hub && python -m src.common.cli finetune --model sd3_5 --data configs/finetune_dataset/dataset.json --images_root configs/finetune_dataset/images --caption_key general_text --out ckpts/sd3_5_lora_stage1 --max_steps 200 --lr 1e-4 --batch_size 1 --grad_accum 4 --seed 42"
```

## 5) Fine-tuned inference (`run_finetuned`)

Use the same venv and Hugging Face cache exports as training. **Stage 1 vs stage 2** only changes `--ckpt`: pass the checkpoint directory that contains `adapters/` (for example `ckpts/pixart_sigma_lora_stage1` after stage 1, or `ckpts/pixart_sigma_lora_stage2` after stage 2).

`ssh_remote.sh` and `start_tmux.sh` already `cd` to `RUNPOD_REMOTE_DIR` before running your command.

### From local machine (SSH wrapper)

```bash
# After sync (e.g. ./scripts/runpod/sync_changed_to_runpod.sh)

# Stage 1 LoRA
./scripts/runpod/ssh_remote.sh "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ImageGenerator/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ImageGenerator/.hf_cache/hub && python -m src.common.cli run_finetuned --model pixart_sigma --ckpt ckpts/pixart_sigma_lora_stage1 --prompt_file configs/prompt_general_clevr.jsonl --mode general --seed 42"

# Stage 2 LoRA
./scripts/runpod/ssh_remote.sh "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ImageGenerator/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ImageGenerator/.hf_cache/hub && python -m src.common.cli run_finetuned --model pixart_sigma --ckpt ckpts/pixart_sigma_lora_stage2 --prompt_file configs/prompt_general_clevr.jsonl --mode general --seed 42"
```

Adjust `--model`, `--ckpt`, `--prompt_file`, and `--mode` (`general` or `general_specific`) for your setup. Images and JSON metadata are written under `outputs/` on the pod (see `configs/experiment.yaml`).

### Optional: inference in tmux

Useful for large JSONL prompt files or if you want logs in `logs/<session>.log`.

```bash
./scripts/runpod/start_tmux.sh pixart_infer_s2 "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ImageGenerator/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ImageGenerator/.hf_cache/hub && python -m src.common.cli run_finetuned --model pixart_sigma --ckpt ckpts/pixart_sigma_lora_stage2 --prompt_file configs/prompt_general_clevr.jsonl --mode general --seed 42"
```

Then: `./scripts/runpod/tail_log.sh pixart_infer_s2`

### Prompts vs fine-tuning

If you trained on CLIP-77 compact captions (`configs/finetune_dataset`), use the same compact style at inference; see `configs/finetune_dataset/README.md` and `scripts/prompt_to_compact.py`.

### Base model (no LoRA)

Use `run`, not `run_finetuned`:

```bash
./scripts/runpod/ssh_remote.sh "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ImageGenerator/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ImageGenerator/.hf_cache/hub && python -m src.common.cli run --model pixart_sigma --prompt_file configs/prompt_general_clevr.jsonl --mode general --seed 42"
```

## 6) Monitor and reconnect

```bash
./scripts/runpod/list_tmux.sh
./scripts/runpod/tail_log.sh pixart_stage1
./scripts/runpod/attach_tmux.sh pixart_stage1
```

Detach from tmux without stopping job: `Ctrl+b`, then `d`.

### Track fine-tuning progress precisely

Use these while a job is running (example session: `pixart_stage1`):

```bash
# Live stream the tmux log
./scripts/runpod/tail_log.sh pixart_stage1

# Check tmux sessions and verify job is still alive
./scripts/runpod/list_tmux.sh

# Read last training records (step/loss/lr) written by the trainer
./scripts/runpod/ssh_remote.sh "python - <<'PY'
from pathlib import Path
p = Path('ckpts/pixart_sigma_lora_stage1/train_log.jsonl')
if not p.exists():
    print('train_log.jsonl not created yet')
else:
    lines = p.read_text().splitlines()
    print(f'total logged updates: {len(lines)}')
    for line in lines[-5:]:
        print(line)
PY"

# Quick current step from latest log entry
./scripts/runpod/ssh_remote.sh "python - <<'PY'
import json
from pathlib import Path
p = Path('ckpts/pixart_sigma_lora_stage1/train_log.jsonl')
if not p.exists() or p.stat().st_size == 0:
    print('no step logged yet')
else:
    last = json.loads(p.read_text().splitlines()[-1])
    print('current_step:', last.get('step'))
    print('latest_loss:', last.get('loss'))
    print('latest_lr:', last.get('lr'))
PY"
```

Interpretation:
- `current_step` grows until it reaches your `--max_steps`.
- The number of lines in `train_log.jsonl` can be lower than `max_steps` when `--grad_accum > 1` (log is written on optimizer steps).

## 7) Pull state back to local

```bash
./scripts/runpod/sync_from_runpod.sh
```

This mirrors remote -> local (including `ckpts/` and `outputs/`), while preserving local `.git/`, `.venv/`, `.env/`.

### Pull a specific folder (recommended script)

```bash
# Merge remote ckpts into local ckpts (recommended)
./scripts/runpod/sync_folder_from_runpod.sh --folder ckpts

# Optional: delete local ckpts entries that do not exist on remote
./scripts/runpod/sync_folder_from_runpod.sh --folder ckpts --delete

# Pull outputs folder
./scripts/runpod/sync_folder_from_runpod.sh --folder outputs
```

One-off `rsync` equivalent (same settings as other scripts):

```bash
# shellcheck source=/dev/null
source scripts/runpod/_common.sh scripts/runpod/runpod.env
mkdir -p ckpts
rsync -az --info=progress2 --no-owner --no-group \
  -e "${RSYNC_SSH}" \
  "${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_REMOTE_DIR}/ckpts/" \
  "./ckpts/"
```

## 8) Sync behavior and safety

- `sync_to_runpod.sh`: mirror local -> remote, uses `--delete`.
- `sync_from_runpod.sh`: mirror remote -> local, uses `--delete`.
- `sync_changed_to_runpod.sh`: incremental push of changed files.

Use mirror commands carefully: whichever side you sync from becomes source of truth.

## 9) Quick troubleshooting

- `Connection refused`: pod host/port changed; update `runpod.env`.
- `Host key verification failed`: remove stale host key then reconnect:
  - `ssh-keygen -R "[${RUNPOD_HOST}]:${RUNPOD_PORT}"`
  - re-run `./scripts/runpod/ssh_remote.sh "hostname"`
- `rsync: command not found`: install rsync on pod.
- `No such file or directory` for remote dir: check `RUNPOD_REMOTE_DIR` points to mounted volume path.
- `.venv/bin/activate: No such file or directory`: run `./scripts/runpod/bootstrap_remote.sh` first.
