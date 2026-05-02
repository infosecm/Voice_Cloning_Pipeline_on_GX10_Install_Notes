# Voice Cloning Pipeline on Asus Ascent GX10 (NVIDIA GB10 / CUDA 13.0 / aarch64)

<!-- Hardware & Platform -->
![NVIDIA GB10](https://img.shields.io/badge/hardware-NVIDIA%20GB10-76b900?logo=nvidia)
![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-76b900?logo=nvidia)
![aarch64](https://img.shields.io/badge/arch-aarch64%20%2F%20arm64-orange)
![Ubuntu 24.04](https://img.shields.io/badge/OS-Ubuntu%2024.04-E95420?logo=ubuntu&logoColor=white)

<!-- Software Stack -->
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange?logo=gradio)
![Local AI](https://img.shields.io/badge/AI-local%20%2F%20self--hosted-success)

<!-- Capabilities -->
![TTS](https://img.shields.io/badge/TTS-text--to--speech-blue)
![Voice Cloning](https://img.shields.io/badge/voice-cloning-purple)
![License](https://img.shields.io/badge/license-various-lightgrey)

<!-- Engines -->
![Fish Speech](https://img.shields.io/badge/engine-Fish%20Speech%201.5-blue)
![Qwen3-TTS](https://img.shields.io/badge/engine-Qwen3--TTS-red)
![VibeVoice](https://img.shields.io/badge/engine-VibeVoice-blueviolet)
![Voxtral](https://img.shields.io/badge/engine-Voxtral%20TTS-ff7000)
![F5-TTS](https://img.shields.io/badge/engine-F5--TTS-informational)
![CosyVoice](https://img.shields.io/badge/engine-CosyVoice%202-success)


Local voice cloning and synthesis pipeline running entirely on the ASUS Ascent GX10 (NVIDIA GB10 Grace Blackwell Superchip). Comparative results table and documented installation notes testing six (6) multilingual synthesis and/or zero-shot voice cloning models: F5-TTS, CosyVoice 2, Fish Speech 1.5, VibeVoice, Qwen3-TTS, and Voxtral TTS.

---

## Table of Contents

- [System Specifications](#system-specifications)
- [GX10-Specific Considerations](#gx10-specific-considerations)
- [Comparative Results Table](#comparative-results-table)
- [Directory Structure on the GX10](#directory-structure-on-the-gx10)
- [Prerequisites](#prerequisites)
- [Virtual Environment Strategy](#virtual-environment-strategy)
- [Installation notes — F5-TTS](#installation--f5-tts)
- [Installation notes — CosyVoice 2](#installation--cosyvoice-2)
- [Installation notes — Fish Speech 1.5](#installation--fish-speech-15)
- [Installation notes — VibeVoice](#installation--vibevoice)
- [Installation notes — Qwen3-TTS](#installation--qwen3-tts)
- [Installation notes — Voxtral TTS](#installation--voxtral-tts)
- [Scripts Usage](#scripts-usage)
- [Audio Reference Files](#audio-reference-files)
- [Troubleshooting](#troubleshooting)

---

## System Specifications

| Component | Details |
|-----------|---------|
| Machine | ASUS Ascent GX10 |
| Chip | NVIDIA GB10 Grace Blackwell Superchip |
| CPU | ARM Cortex-X925 (10 cores, 4.0 GHz) + Cortex-A725 (10 cores, 2.86 GHz) |
| Architecture | aarch64 (ARM64) |
| Unified RAM | 128 GB LPDDR5 @ 8533 MT/s (shared CPU + GPU) |
| GPU | NVIDIA GB10 (Blackwell, compute capability 12.1) |
| CUDA | 13.0 |
| OS | Ubuntu 24.04.3 LTS (Noble) |
| Kernel | 6.11.0-nvidia (aarch64) |
| Python | 3.12.3 |
| Storage | 1, 2 or 4 TB NVMe |

---

## GX10-Specific Considerations

The GX10 has several important differences from a standard x86 Linux workstation that affect installation significantly.

**Unified memory architecture.** The GPU does not have dedicated VRAM — it shares the full 128 GB with the CPU. `nvidia-smi` reports `N/A` for memory usage, which is normal. All models load directly into unified memory.

**ARM aarch64.** Many Python packages do not provide prebuilt `aarch64` wheels. Several packages must be compiled from source or pinned to specific versions that provide aarch64 wheels. This is the primary source of installation friction throughout this guide.

**CUDA 13.0 + PyTorch.** The GX10 ships with CUDA 13.0. As of mid-2025, there are no official PyTorch wheels for CUDA 13.0 on aarch64 published on PyPI or `download.pytorch.org`. The CUDA-enabled PyTorch (`torch 2.11.0+cu130`) must be installed manually from the NVIDIA Python Package Index into a dedicated base venv (`~/nvidia_torch_venv/`) before proceeding with any engine installation. See the [prerequisites](#prerequisites) section below.

**Torch version pinning.** Several packages attempt to install their own version of PyTorch during setup, which silently overwrites the CUDA-enabled version with a CPU-only build from PyPI. The fix is to always restore the `.pth` pointer to the dedicated torch base venv after installing any package that depends on torch.

**ROCm not available.** The machine uses NVIDIA drivers (580.x). Ignore any ROCm-related instructions.

**flash-attn.** A prebuilt wheel for aarch64 is available in the pip cache once installed for one venv. For subsequent venvs, install directly from the cached wheel to avoid recompilation:
```bash
find ~/.cache/pip -name "flash_attn*.whl" 2>/dev/null
pip install /path/to/cached/flash_attn-*.whl
```

---

## Comparative Results Table

| # | Engine | Languages | Voice Cloning | Multi-speaker | Max Length | Interface | Best For |
|---|--------|-----------|---------------|---------------|------------|-----------|----------|
| 1 | Qwen3-TTS 1.7B | 10 | ✅ Zero-shot (3s ref) | ❌ | 10 min/segment | Gradio :7863 | Best overall — Fine-tuning, long-form |
| 2 | Fish Speech 1.5 | 50+ | ✅ Zero-shot | ❌ | Batch | CLI script | Best multilingual cloning |
| 3 | VibeVoice 1.5B | EN/ZH | ✅ Zero-shot | ✅ 4 speakers | 90 min | Gradio :7862 | Best for English podcasts, multi-voice |
| 4 | Voxtral TTS | 9 | ⚠️ Preset voices only | ❌ | ~2m40s/segment | CLI script | No Python required, solid multilingual presets |
| 5 | F5-TTS | EN primary | ✅ Zero-shot | ❌ | Batch | Gradio :7861 | English cloning, rapid prototyping |
| 6 | CosyVoice 2 | ZH/EN | ✅ 30s ref max | ❌ | Batch | CLI script | Chinese synthesis |

> **Best overall:** Qwen3-TTS 1.7B — fine-tunable, excellent French and English, long-form capable  
> **Best multilingual:** Fish Speech 1.5 — 50+ languages, true zero-shot, no ref transcript needed  
> **Best for podcasts:** VibeVoice 1.5B — only engine supporting 4 simultaneous speakers  
> **No Python needed:** Voxtral TTS — pure C binary, no venv, no dependencies  

---

## Directory Structure on the GX10

```
~/voice_clone/
├── audio_samples/                  # Reference audio files and generated outputs
│   ├── ref_en_25s.wav              # Your self-recorded reference for VibeVoice/Qwen3
│   └── output_*.wav                # Generated audio files (timestamped)
├── CosyVoice/                      # CosyVoice source (git clone)
├── cosyvoice_clone.sh              # CosyVoice 2 pipeline script
├── f5tts/                          # F5-TTS dedicated venv (gradio 6.x)
│   ├── bin/
│   ├── lib/
│   └── ...
├── fish-speech/                    # Fish Speech source (git clone, tag v1.5.1)
├── fish_speech_clone.sh            # Fish Speech 1.5 pipeline script
├── qwen3tts/                       # Qwen3-TTS source + venv
│   ├── .venv/                      # Dedicated venv
│   ├── finetuning/                 # Fine-tuning dataset and output
│   │   ├── dataset/
│   │   │   └── audio/              # WAV segments for fine-tuning
│   │   └── output/                 # Fine-tuned model output
│   ├── models/                     # Downloaded model weights
│   │   ├── Qwen3-TTS-12Hz-1.7B-Base/        # ~4.5 GB — recommended for voice cloning
│   │   ├── Qwen3-TTS-12Hz-1.7B-VoiceDesign/ # Optional
│   │   └── Qwen3-TTS-Tokenizer-12Hz/
│   ├── output_audio/               # Generated audio files
│   └── src/                        # Source repo (git clone)
├── start_f5tts_gradio.sh           # F5-TTS Gradio web UI launcher
├── start_qwen3tts_gradio.sh        # Qwen3-TTS Gradio web UI launcher
├── start_vibevoice_gradio.sh       # VibeVoice Gradio web UI launcher
├── vibevoice/                      # VibeVoice source + venv
│   ├── .venv/                      # Dedicated venv
│   ├── demo/                       # Gradio demo scripts
│   │   └── voices/                 # Custom voice WAV files for speaker selection
│   └── models/
│       └── VibeVoice-1.5B/        # Model weights (~6 GB)
├── voxtral/                        # Voxtral TTS (pure C implementation)
│   ├── voxtral_tts                 # Compiled binary
│   └── voxtral-tts-model/          # Model weights (~8 GB)
└── voxtral_generate.py             # Voxtral TTS Python wrapper script

~/nvidia_torch_venv/                # Dedicated torch base venv
```

---

## Prerequisites

### System packages

```bash
sudo apt-get install -y \
    espeak-ng \
    ffmpeg \
    sox \
    python3-dev \
    portaudio19-dev
```

> **Note on `git-lfs`:** The Ubuntu ESM repository has a broken entry for `git-lfs` on aarch64. Install the binary directly instead:
> ```bash
> curl -L https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-arm64-v3.5.1.tar.gz \
>     -o /tmp/git-lfs.tar.gz
> cd /tmp && tar -xzf git-lfs.tar.gz
> sudo ./git-lfs-3.5.1/install.sh
> git lfs version  # should print git-lfs/3.5.1
> ```


### Install torch 2.11.0+cu130 into a dedicated venv

> **GX10 / DGX Spark users:** `torch 2.11.0+cu130` for aarch64 is not available on PyPI. You must install it from the NVIDIA Python Package Index. This step is required before creating any venv in this guide.

```bash
# Create the NVIDIA torch base venv
python3 -m venv ~/nvidia_torch_venv

source ~/nvidia_torch_venv/bin/activate

# Install torch from the NVIDIA package index
pip install --upgrade pip
pip install torch==2.11.0 \
    --index-url https://pypi.nvidia.com/cu130/

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.11.0+cu130 True

deactivate
```

---

## Virtual Environment Strategy

Five venvs are used to avoid dependency conflicts:

| Venv | Path | Purpose | Gradio version |
|------|------|---------|----------------|
| `voice_clone` | `~/voice_clone/` | CosyVoice 2 + Fish Speech | 5.4.0 |
| `f5tts` | `~/voice_clone/f5tts/` | F5-TTS only | 6.10.0 |
| `vibevoice` | `~/voice_clone/vibevoice/.venv/` | VibeVoice | — |
| `qwen3tts` | `~/voice_clone/qwen3tts/.venv/` | Qwen3-TTS | — |
| NVIDIA base | `~/nvidia_torch_venv/` | Source of torch 2.11+cu130 | — |

All venvs inherit torch from the NVIDIA base environment via a `.pth` file.

### Create the main voice_clone venv

```bash
python3 -m venv ~/voice_clone

# Point to the NVIDIA-provided torch
echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/lib/python3.12/site-packages/nvidia_torch.pth

source ~/voice_clone/bin/activate

# Verify CUDA is available
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.11.0+cu130 True
```

### Create the f5tts venv

```bash
python3 -m venv ~/voice_clone/f5tts

echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/f5tts/lib/python3.12/site-packages/nvidia_torch.pth
```

---

## Installation — F5-TTS

F5-TTS is a zero-shot voice cloning model. It works well in English but may produce noticeable accent artifacts in other languages (ex.: a Scandinavian-sounding accent in French). Best used for English synthesis or rapid prototyping.

### Install

```bash
source ~/voice_clone/f5tts/bin/activate

pip install f5-tts
pip install "gradio>=6.0.0,<6.11"

# Fix wandb conflict: the NVIDIA base venv has an old wandb that uses distutils
# (removed in Python 3.12). Force-install a newer version in this venv.
~/voice_clone/f5tts/bin/pip install wandb --upgrade --force-reinstall --no-deps

# Restore NVIDIA torch
echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/f5tts/lib/python3.12/site-packages/nvidia_torch.pth

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.11.0+cu130 True
python -c "from f5_tts.api import F5TTS; print('F5-TTS OK')"
```

### Quick test

```bash
python -m f5_tts.infer.infer_gradio -H 0.0.0.0 -p 7861
```

### Launch Gradio UI

```bash
~/voice_clone/start_f5tts_gradio.sh
# Access at http://<GX10_IP>:7861
```

### Launch script — `start_f5tts_gradio.sh`

```bash
#!/bin/bash
echo "🎙️ Starting F5-TTS on Gradio..."

echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/f5tts/lib/python3.12/site-packages/nvidia_torch.pth

source ~/voice_clone/f5tts/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('✅ CUDA OK:', torch.__version__)"

trap 'deactivate; exit' INT TERM

python -m f5_tts.infer.infer_gradio -H 0.0.0.0 -p 7861
deactivate
```

---

## Installation — CosyVoice 2

CosyVoice 2 is a multilingual TTS model from Alibaba. It supports other languages but with a noticeable English accent. Best suited for Chinese or English synthesis; other languages may not sound fully natural.

### Install

```bash
source ~/voice_clone/bin/activate

# Clone repository
cd ~/voice_clone
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

# --- Patch requirements.txt before installing ---

# 1. Remove onnxruntime-gpu: no aarch64 wheel exists
sed -i 's/onnxruntime-gpu==1.18.0//' requirements.txt
sed -i '/^$/d' requirements.txt
sed -i '/^; sys_platform/d' requirements.txt

# 2. grpcio 1.57.0 has no aarch64 wheel and fails to compile
sed -i 's/grpcio==1.57.0/grpcio==1.67.0/' requirements.txt
sed -i 's/grpcio-tools==1.57.0/grpcio-tools==1.66.0/' requirements.txt

# 3. protobuf conflict with grpcio-tools 1.66
sed -i 's/protobuf==4.25/protobuf==5.29.6/' requirements.txt

# 4. tensorrt 10.13.3.9 does not exist on aarch64
sed -i 's/tensorrt-cu12==10.13.3.9/tensorrt-cu12==10.13.0.35/' requirements.txt
sed -i 's/tensorrt-cu12-bindings==10.13.3.9/tensorrt-cu12-bindings==10.13.0.35/' requirements.txt
sed -i 's/tensorrt-cu12-libs==10.13.3.9/tensorrt-cu12-libs==10.13.0.35/' requirements.txt

# --- Install ---

# openai-whisper must be built without pip isolation (setuptools 82 issue)
pip install openai-whisper==20231117 --no-build-isolation

# Install remaining requirements
pip install -r requirements.txt --no-build-isolation

# onnxruntime: GPU version unavailable for aarch64, use CPU version
pip install onnxruntime

# Add CosyVoice to Python path
echo "~/voice_clone/CosyVoice" \
    > ~/voice_clone/lib/python3.12/site-packages/cosyvoice.pth
echo "~/voice_clone/CosyVoice/third_party/Matcha-TTS" \
    >> ~/voice_clone/lib/python3.12/site-packages/cosyvoice.pth

# Install torchvision compatible with torch 2.11
pip install torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130

# Restore NVIDIA torch (CosyVoice install overwrites it with a CPU-only version)
echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/lib/python3.12/site-packages/nvidia_torch.pth

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.11.0+cu130 True
python -c "import cosyvoice; print('CosyVoice OK')"
```

### Download model

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='FunAudioLLM/CosyVoice2-0.5B',
    local_dir='~/voice_clone/CosyVoice/pretrained_models/CosyVoice2-0.5B'
)
print('Model downloaded OK')
"
```

### Quick test

```bash
cd ~/voice_clone/CosyVoice
python -c "
import torch, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cosyvoice.cli.cosyvoice import CosyVoice2
import torchaudio

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False)
print('Device:', next(cosyvoice.model.flow.parameters()).device)

output = cosyvoice.inference_zero_shot(
    'Hello, this is a voice cloning test on the Asus Ascent GX10.',
    'Some call me nature, others call me mother nature.',
    '~/voice_clone/audio_samples/ref_en_25s.wav',
    stream=False
)
for i, result in enumerate(output):
    torchaudio.save(f'/tmp/test_cosyvoice_{i}.wav', result['tts_speech'], cosyvoice.sample_rate)
    print(f'Generated: /tmp/test_cosyvoice_{i}.wav')
"
```

> **Note:** The reference audio must be 30 seconds or less. CosyVoice enforces this limit strictly.

---

## Installation — Fish Speech 1.5

Based on our test results, Fish Speech produces natural-sounding voices without foreign accent artifacts, unlike F5-TTS and CosyVoice 2.

### Install

```bash
source ~/voice_clone/bin/activate

cd ~/voice_clone
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Use the version matching the checkpoint
git checkout v1.5.1

# Install Fish Speech
pip install -e .

# Restore NVIDIA torch (Fish Speech install overwrites it)
echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/lib/python3.12/site-packages/nvidia_torch.pth

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.11.0+cu130 True
```

### Download model

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='fishaudio/fish-speech-1.5',
    local_dir='~/voice_clone/fish-speech/checkpoints/fish-speech-1.5'
)
print('Model downloaded OK')
"
```

### Quick test (3-step pipeline)

```bash
cd ~/voice_clone/fish-speech
CHECKPOINT="checkpoints/fish-speech-1.5"
REF_AUDIO="~/voice_clone/audio_samples/ref_en_25s.wav"
SAMPLES="~/voice_clone/audio_samples"

# Step 1: Encode reference audio to VQ tokens
python fish_speech/models/vqgan/inference.py \
    -i "$REF_AUDIO" \
    --checkpoint-path "$CHECKPOINT/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    -o /tmp/ref_tokens_tmp.wav \
    -d cuda

# Step 2: Generate semantic codes from text
python fish_speech/models/text2semantic/inference.py \
    --text "Hello, this is a voice cloning test on the Asus Ascent GX10." \
    --prompt-text "Some call me nature, others call me mother nature." \
    --prompt-tokens "$SAMPLES/ref_en_25s.npy" \
    --checkpoint-path "$CHECKPOINT" \
    --output-dir "$SAMPLES" \
    --device cuda \
    --num-samples 1

# Step 3: Decode codes to audio
python fish_speech/models/vqgan/inference.py \
    -i "$SAMPLES/codes_0.npy" \
    --checkpoint-path "$CHECKPOINT/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    -o /tmp/test_fish_speech.wav \
    -d cuda
```

> **Note on prompt text:** Avoid apostrophes and accented characters when passing the reference transcript as a shell argument — they cause quoting errors. Use simplified ASCII text instead.

---

## Installation — VibeVoice

VibeVoice is a Microsoft open-source TTS model optimized for **English podcast generation with up to 4 distinct speakers** and up to 90 minutes of continuous audio in a single pass. It supports voice cloning.

The official Microsoft repo had its TTS code removed in September 2025. Use the community fork: `github.com/vibevoice-community/VibeVoice`.

### Install

```bash
cd ~/voice_clone

# Clone repo (venv lives inside the source directory)
git clone https://github.com/vibevoice-community/VibeVoice.git vibevoice
cd vibevoice

# Create venv inside the repo
python3 -m venv .venv

# Point to NVIDIA torch
echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > .venv/lib/python3.12/site-packages/nvidia_torch.pth

source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.11.0+cu130 True

# Install
pip install uv
uv pip install -e .

# Restore NVIDIA torch
echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > .venv/lib/python3.12/site-packages/nvidia_torch.pth

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.11.0+cu130 True
```

### Install flash-attn (required)

VibeVoice requires flash-attn. A prebuilt aarch64 wheel is available in the pip cache after the first installation (e.g. for Qwen3-TTS). Install from cache to avoid recompilation:

```bash
# Find the cached wheel
find ~/.cache/pip -name "flash_attn*.whl" 2>/dev/null

# Install directly from cache
pip install ~/.cache/pip/wheels/.../flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl

# If no cache exists yet, compile from source (takes 20-30 minutes)
pip install wheel
pip install flash-attn --no-build-isolation
```

### Fix Gradio to listen on all interfaces

By default the Gradio demo only listens on localhost. Edit the launch configuration:

```bash
# Enable port argument
sed -i 's/# server_port=args.port,/server_port=args.port,/' demo/gradio_demo.py

# Listen on all interfaces (not just localhost)
sed -i 's/server_name="0.0.0.0" if args.share else "127.0.0.1"/server_name="0.0.0.0"/' demo/gradio_demo.py
```

### Add your voice to the available voices

Place your reference WAV in the voices directory — it will appear in the Gradio speaker dropdown:

```bash
cp ~/voice_clone/audio_samples/ref_en_25s.wav \
    ~/voice_clone/vibevoice/demo/voices/en-BobSmith_man.wav
```

### Download model

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='vibevoice/VibeVoice-1.5B',
    local_dir='~/voice_clone/vibevoice/models/VibeVoice-1.5B'
)
print('Model downloaded.')
"
```

### Test without voice cloning

```bash
cd ~/voice_clone/vibevoice
python demo/inference_from_file.py \
    --model_path models/VibeVoice-1.5B \
    --txt_path demo/text_examples/1p_abs.txt \
    --speaker_names Alice \
    --disable_prefill
```

### Launch Gradio UI

```bash
~/voice_clone/start_vibevoice.sh
# Access at http://<GX10_IP>:7862
```

### Launch script — `start_vibevoice_gradio.sh`

```bash
#!/bin/bash
echo "🎙️ Starting VibeVoice on Gradio..."

echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/vibevoice/.venv/lib/python3.12/site-packages/nvidia_torch.pth

source ~/voice_clone/vibevoice/.venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('✅ CUDA OK:', torch.__version__)"

trap 'deactivate; exit' INT TERM

cd ~/voice_clone/vibevoice
python demo/gradio_demo.py --model_path models/VibeVoice-1.5B --port 7862
deactivate
```

### Multi-speaker podcast script format

```
Speaker 1: Welcome to today's show. We have an interesting topic lined up for you.
Speaker 2: That's right. Let's dive straight into the details.
Speaker 1: Our first segment covers the latest developments in the field.
```

---

## Installation — Qwen3-TTS

Qwen3-TTS produces excellent voice quality, supports voice cloning from a 3-second reference, and handles up to 10 minutes of continuous generation per segment. Apache 2.0 license.

Three model variants are available. Use **1.7B-Base** for voice cloning with your own reference audio.

| Model | Use case | Size |
|-------|----------|------|
| `Qwen3-TTS-12Hz-1.7B-Base` | Voice cloning from reference audio | ~4.5 GB |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Generate voice from text description | ~4.5 GB |
| `Qwen3-TTS-12Hz-0.6B-Base` | Lighter, but degrades on long-form | ~2.5 GB |

> **Note:** Do NOT use 0.6B-Base for long-form voice cloning — it produces excessive silences and pauses on texts longer than a few sentences.

### Install

```bash
# Create dedicated directory and venv
mkdir -p ~/voice_clone/qwen3tts/{models,finetuning/dataset/audio,finetuning/output,output_audio,src}
python3 -m venv ~/voice_clone/qwen3tts/.venv

# Point to NVIDIA torch
echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/qwen3tts/.venv/lib/python3.12/site-packages/nvidia_torch.pth

source ~/voice_clone/qwen3tts/.venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.11.0+cu130 True

# Install qwen-tts
pip install -U qwen-tts

# Install flash-attn from cache (required — see VibeVoice section for first-time compilation)
pip install ~/.cache/pip/wheels/.../flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl

# Install sox (required for audio processing)
sudo apt install sox -y

# Restore NVIDIA torch
echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/qwen3tts/.venv/lib/python3.12/site-packages/nvidia_torch.pth

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.11.0+cu130 True
```

### Download models

```bash
source ~/voice_clone/qwen3tts/.venv/bin/activate

# Tokenizer (required for all models)
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --local-dir ~/voice_clone/qwen3tts/models/Qwen3-TTS-Tokenizer-12Hz

# Base model for voice cloning (recommended)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --local-dir ~/voice_clone/qwen3tts/models/Qwen3-TTS-12Hz-1.7B-Base

# VoiceDesign model (optional)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --local-dir ~/voice_clone/qwen3tts/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

### Quick test — voice cloning in English

```bash
source ~/voice_clone/qwen3tts/.venv/bin/activate

python - << 'EOF'
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "~/voice_clone/qwen3tts/models/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

ref_audio = "~/voice_clone/audio_samples/ref_en_25s.wav"
ref_text = "Hello. Today I will talk to you about a fascinating subject: artificial intelligence and its impact on our daily lives."

wavs, sr = model.generate_voice_clone(
    text="So, the technology landscape is evolving rapidly across every industry sector.",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
    max_new_tokens=4096,
)
sf.write("~/voice_clone/audio_samples/test_qwen3.wav", wavs[0], sr)
print("Audio generated: test_qwen3.wav")
EOF
```

> **Known issue — first token cut off:** Qwen3-TTS sometimes skips the first word of the generated text. Workaround: prefix your text with a sacrificial word (e.g. `"So, ..."`) and trim it in post-production with ffmpeg.

### Launch Gradio UI

```bash
~/voice_clone/start_qwen3tts.sh
# Access at http://<GX10_IP>:7863
```

### Launch script — `start_qwen3tts_gradio.sh`

```bash
#!/bin/bash
echo "🎙️ Starting Qwen3-TTS on Gradio..."

echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/qwen3tts/.venv/lib/python3.12/site-packages/nvidia_torch.pth

source ~/voice_clone/qwen3tts/.venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('✅ CUDA OK:', torch.__version__)"

trap 'deactivate; exit' INT TERM

qwen-tts-demo ~/voice_clone/qwen3tts/models/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 7863 --dtype bfloat16
deactivate
```

### Fine-tuning your voice (optional)

Fine-tuning permanently bakes your voice into the model weights, eliminating the need for a reference audio clip at inference time.

**Dataset requirements:**
- Minimum 30 minutes of clean audio, ideally 1–2 hours
- WAV, 24 kHz, mono, segments of 5–30 seconds each
- Exact transcript for each segment

**Dataset format (`finetuning/dataset/metadata.jsonl`):**
```json
{"audio": "audio/segment_001.wav", "text": "Text to read in the first segment."}
{"audio": "audio/segment_002.wav", "text": "Text to read in the second segment."}
```

A phonetically balanced corpus of about 150 sentences is highly recommended.

**Fine-tuning command:**
```bash
source ~/voice_clone/qwen3tts/.venv/bin/activate
cd ~/voice_clone/qwen3tts/src

python finetune.py \
    --model_path ~/voice_clone/qwen3tts/models/Qwen3-TTS-12Hz-1.7B-Base \
    --dataset_path ~/voice_clone/qwen3tts/finetuning/dataset \
    --output_dir ~/voice_clone/qwen3tts/finetuning/output \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --batch_size 4
```

The fine-tuned model is used identically to the base model — simply point `model_path` to the output directory.

---

## Installation — Voxtral TTS

Voxtral TTS is Mistral's open-weight multilingual TTS model released March 26, 2026. It supports 9 languages natively, with 20 preset voices and 24 kHz audio output. Blind human evaluations show 62.8% preference over ElevenLabs Flash v2.5.

### Architecture

Voxtral TTS is a 4B parameter model combining a Ministral 3B LLM backbone with a flow-matching acoustic transformer and an audio codec decoder. It generates 37 audio tokens per frame (1 semantic + 36 acoustic).

### Current limitation on GX10 — vLLM-Omni not available

The official serving stack for Voxtral TTS is **vLLM-Omni** (>= 0.18.0), which provides a persistent server, Gradio UI, streaming inference, and sub-100ms latency. However, vLLM-Omni is **not currently installable on aarch64 + CUDA 13.0** for the following reasons:

- No prebuilt aarch64 wheel exists for vLLM or vLLM-Omni
- vLLM compiled for CUDA 12.x is incompatible with CUDA 13.x (not backward compatible)
- ARM64 Docker image support for vLLM-Omni is an open feature request (December 2025, no response)
- Building from source requires CUDA 12.x, which cannot coexist with the GX10's CUDA 13.0 stack

The alternative is the **pure C implementation** (`voxtral-tts.c` by antirez), which has been tested specifically on the NVIDIA GB10 Blackwell with 128 GB unified memory. It runs on CPU via OpenBLAS with no Python dependencies and compiles in seconds. Limitations: no persistent server (model reloads on each call, ~40s), CLI only, no Gradio UI, no streaming.

**When vLLM-Omni gains official ARM64 support**, the serving approach in these installation notes may be revisited.

### Install dependencies

```bash
sudo apt install libopenblas-dev -y
```

### Compile

```bash
mkdir -p ~/voice_clone/voxtral
cd ~/voice_clone/voxtral
git clone https://github.com/mudler/voxtral-tts.c .
make
```

The compilation produces a single binary `voxtral_tts` with no runtime Python dependencies.

### Download model weights

```bash
cd ~/voice_clone/voxtral
./download_model.sh voxtral-tts-model
```

Model size: ~8 GB. Downloaded to `~/voice_clone/voxtral/voxtral-tts-model/`.

### Available voices

```
English : casual_female, casual_male, cheerful_female, neutral_female, neutral_male
French  : fr_female, fr_male
German  : de_female, de_male
Spanish : es_female, es_male
Italian : it_female, it_male
Portuguese: pt_female, pt_male
Dutch   : nl_female, nl_male
Arabic  : ar_male
Hindi   : hi_female, hi_male
```

> **Important:** Always specify a language-appropriate voice. Using `neutral_female` (English) for other languages produces a heavy English accent. For instance, use `fr_female` or `fr_male` for French.

### Quick test

```bash
cd ~/voice_clone/voxtral

./voxtral_tts \
    -d voxtral-tts-model \
    -v neutral_female \
    -o ~/voice_clone/audio_samples/test_voxtral_en.wav \
    "So, the technology landscape is evolving rapidly across every industry sector."
```

> **Known issue — first syllable cut off:** Voxtral sometimes skips the first syllable. Workaround: prefix the text with a sacrificial word (`"So, ..."`) and trim in post-production with ffmpeg.

```bash
ffmpeg -i ~/voice_clone/audio_samples/test_voxtral_fr.wav \
    -af silenceremove=start_periods=1:start_threshold=-50dB \
    ~/voice_clone/audio_samples/test_voxtral_fr_trimmed.wav
```

### Python wrapper script — `voxtral_generate.py`

Since the binary reloads the model on each call (~40s), the Python wrapper below allows batching multiple lines of text into a single generation call, then assembles the segments with ffmpeg.

```bash
cat > ~/voice_clone/voxtral_generate.py << 'EOF'
#!/usr/bin/env python3
"""
Voxtral TTS — Python wrapper for batch generation.
Usage:
    python voxtral_generate.py "Your text here." [options]
    python voxtral_generate.py --file script.txt [options]

Options:
    --voice     Voice name (default: fr_female)
    --output    Output WAV file (default: ~/voice_clone/audio_samples/output_voxtral_TIMESTAMP.wav)
    --file      Read text from file instead of argument
    --no-trim   Skip silence trimming
"""

import argparse
import subprocess
import os
import sys
import datetime

VOICE_CLONE_DIR = os.path.expanduser("~/voice_clone")
BINARY = os.path.join(VOICE_CLONE_DIR, "voxtral", "voxtral_tts")
MODEL_DIR = os.path.join(VOICE_CLONE_DIR, "voxtral", "voxtral-tts-model")
SAMPLES_DIR = os.path.join(VOICE_CLONE_DIR, "audio_samples")


def generate(text, voice, output_path, trim=True):
    # Add sacrificial word to avoid first-syllable cutoff
    prefixed_text = "Donc, " + text if not text.startswith("Donc,") else text

    raw_output = output_path.replace(".wav", "_raw.wav")

    cmd = [
        BINARY,
        "-d", MODEL_DIR,
        "-v", voice,
        "-o", raw_output,
        prefixed_text
    ]

    print(f"Generating: {voice} → {os.path.basename(output_path)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"Error: voxtral_tts exited with code {result.returncode}")
        sys.exit(1)

    if trim:
        trim_cmd = [
            "ffmpeg", "-y",
            "-i", raw_output,
            "-af", "silenceremove=start_periods=1:start_threshold=-50dB",
            output_path
        ]
        subprocess.run(trim_cmd, capture_output=True)
        os.remove(raw_output)
    else:
        os.rename(raw_output, output_path)

    duration_cmd = ["ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", output_path]
    try:
        dur = subprocess.check_output(duration_cmd, stderr=subprocess.DEVNULL).decode().strip()
        print(f"Output: {output_path} ({float(dur):.1f}s)")
    except Exception:
        print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Voxtral TTS wrapper")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--voice", default="neutral_female", help="Voice name (default: neutral_female)")
    parser.add_argument("--output", default=None, help="Output WAV file path")
    parser.add_argument("--file", default=None, help="Read text from file")
    parser.add_argument("--no-trim", action="store_true", help="Skip silence trimming")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        sys.exit(1)

    if args.output:
        output_path = os.path.expanduser(args.output)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(SAMPLES_DIR, f"output_voxtral_{timestamp}.wav")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generate(text, args.voice, output_path, trim=not args.no_trim)


if __name__ == "__main__":
    main()
EOF

chmod +x ~/voice_clone/voxtral_generate.py
```

### Usage examples

```bash
cd ~/voice_clone

# English synthesis
python voxtral_generate.py "The technology landscape is evolving rapidly across every industry sector." --voice neutral_female

# Specify voice and output
python voxtral_generate.py "Welcome to today's episode." --voice neutral_male --output ~/voice_clone/audio_samples/intro.wav

# French synthesis
python voxtral_generate.py "Les nouvelles technologies transforment notre façon de travailler." --voice fr_female

# From a text file
python voxtral_generate.py --file /tmp/script.txt --voice fr_female

# List all available voices
./voxtral/voxtral_tts --list-voices
```

---

## Scripts Usage

All scripts are located in `~/voice_clone/`.

---

### `fish_speech_clone.sh` — Fish Speech voice cloning

```bash
# Basic usage — synthesizes text using the default reference voice
./fish_speech_clone.sh "Hello, this is a cloned voice using Fish Speech."

# With a custom reference audio and transcript
./fish_speech_clone.sh "Text" /path/to/ref.wav "Transcript of the reference audio"
```

**Output:** `~/voice_clone/audio_samples/output_YYYYMMDD_HHMMSS.wav`

**Notes:**
- Automatically restores NVIDIA torch 2.11+cu130 before running
- Default reference: `~/voice_clone/audio_samples/ref_en_25s.wav`
- Avoid apostrophes in the reference transcript ($3) — use simplified ASCII text

---

### `cosyvoice_clone.sh` — CosyVoice 2 voice cloning

```bash
./cosyvoice_clone.sh "Hello, this is a cloned voice using CosyVoice 2."
./cosyvoice_clone.sh "Text" /path/to/ref.wav "Reference transcript"
```

**Output:** `~/voice_clone/audio_samples/output_cosyvoice_YYYYMMDD_HHMMSS.wav`

**Notes:**
- Reference audio must be 30 seconds or less
- Automatically restores NVIDIA torch 2.11+cu130

---

### `voxtral_generate.py` — Voxtral TTS generation

```bash
cd ~/voice_clone

# English synthesis
python voxtral_generate.py "The technology landscape is evolving rapidly." --voice neutral_female

# Specify voice
python voxtral_generate.py "Welcome to today's episode." --voice neutral_male

# French synthesis
python voxtral_generate.py "Les nouvelles technologies transforment notre façon de travailler." --voice fr_female

# From a script file
python voxtral_generate.py --file /tmp/script.txt --voice fr_female
```

**Notes:**
- No venv required — pure C binary, no Python dependencies for generation
- Model reloads on each call (~40s) — batch as much text as possible per call
- Always use a language-appropriate voice (ex.: `fr_female`, `fr_male` for French)
- vLLM-Omni (persistent server, Gradio UI) is not yet available for aarch64 + CUDA 13.0

---

```bash
./start_f5tts_gradio.sh
# Access at http://<GX10_IP>:7861
```

---

### `start_vibevoice_gradio.sh` — VibeVoice web interface

```bash
./start_vibevoice_gradio.sh
# Access at http://<GX10_IP>:7862
```

**Notes:**
- Best for English podcast generation with multiple speakers
- Up to 4 distinct voices, up to 90 minutes continuous
- Place reference WAV files in `~/voice_clone/vibevoice/demo/voices/` to add custom voices

---

### `start_qwen3tts_gradio.sh` — Qwen3-TTS web interface

```bash
./start_qwen3tts_gradio.sh
# Access at http://<GX10_IP>:7863
```

**Notes:**
- Supports up to 10 minutes of continuous generation per segment
- For longer content, generate in segments and assemble with ffmpeg
- All scripts use `trap 'deactivate; exit' INT TERM` — CTRL-C cleanly deactivates the venv

---

## Audio Reference Files

For best results, record a reference audio file with these characteristics:

| Property | Recommendation |
|----------|---------------|
| Duration | 25–30 seconds |
| Format | WAV, 24 kHz, mono |
| Environment | Quiet room, no background noise or music |
| Content | Natural speech, varied sentences, questions and statements |
| Language | Must match the language you want to synthesize |

### Preparing a reference file

```bash
# Convert any format to WAV 24kHz mono (25 seconds)
ffmpeg -i input.m4a -t 25 -ar 24000 -ac 1 ~/voice_clone/audio_samples/ref_en_25s.wav

# Transcribe with Whisper on GPU
source ~/voice_clone/bin/activate
python -c "
import whisper
model = whisper.load_model('medium', device='cuda')
result = model.transcribe('~/voice_clone/audio_samples/ref_en_25s.wav', language='fr')
print(result['text'])
"
```

---

## Troubleshooting

### CUDA not available after installing a new package

Any pip package that depends on `torch` will replace the NVIDIA torch with a CPU-only build from PyPI. This is the most common issue. Fix:

```bash
source ~/voice_clone/bin/activate
echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/lib/python3.12/site-packages/nvidia_torch.pth
python -c "import torch; print(torch.cuda.is_available())"  # must print True
```

Apply the same fix for any other venv (vibevoice, qwen3tts, f5tts) by replacing the path accordingly:

```bash
# f5tts
echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/f5tts/lib/python3.12/site-packages/nvidia_torch.pth

# vibevoice
echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/vibevoice/.venv/lib/python3.12/site-packages/nvidia_torch.pth

# qwen3tts
echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/qwen3tts/.venv/lib/python3.12/site-packages/nvidia_torch.pth
```

---

### `onnxruntime-gpu` not found

No aarch64 wheel exists for `onnxruntime-gpu`. Use the CPU version — the main TTS models still run on GPU:

```bash
pip install onnxruntime
```

---

### `grpcio` compilation fails

grpcio 1.57 has no prebuilt aarch64 wheel and fails to compile. Use 1.67:

```bash
sed -i 's/grpcio==1.57.0/grpcio==1.67.0/' requirements.txt
```

---

### `git-lfs` 404 from Ubuntu ESM repo

```bash
curl -L https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-arm64-v3.5.1.tar.gz \
    -o /tmp/git-lfs.tar.gz
cd /tmp && tar -xzf git-lfs.tar.gz
sudo ./git-lfs-3.5.1/install.sh
```

---

### `openai-whisper` fails to build (`pkg_resources` missing)

setuptools 82 removed `pkg_resources`. Build without isolation:

```bash
pip install openai-whisper==20231117 --no-build-isolation
```

---

### Fish Speech codec size mismatch at runtime

Caused by a mismatch between the code version and the checkpoint. Always use git tag `v1.5.1` with the `fish-speech-1.5` checkpoint:

```bash
cd ~/voice_clone/fish-speech
git checkout v1.5.1
pip install -e .
```

---

### F5-TTS Gradio `TypeError: Textbox.__init__() unexpected keyword argument 'buttons'`

F5-TTS requires gradio 6.x but `voice_clone` has gradio 5.x (required by CosyVoice). They cannot share a venv. F5-TTS must run in its own venv:

```bash
python3 -m venv ~/voice_clone/f5tts
echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/f5tts/lib/python3.12/site-packages/nvidia_torch.pth
source ~/voice_clone/f5tts/bin/activate
pip install f5-tts "gradio>=6.0.0,<6.11"
~/voice_clone/f5tts/bin/pip install wandb --upgrade --force-reinstall --no-deps
```

---

### `ModuleNotFoundError: No module named 'distutils'` (wandb)

wandb 0.18 in the NVIDIA base venv uses `distutils`, removed in Python 3.12. Fix by installing a newer wandb directly in the target venv:

```bash
~/voice_clone/f5tts/bin/pip install wandb --upgrade --force-reinstall --no-deps
```

---

### CosyVoice `AssertionError: do not support extract speech token for audio longer than 30s`

CosyVoice enforces a hard 30-second limit on reference audio. Trim with ffmpeg:

```bash
ffmpeg -i ref_audio.wav -t 25 -ar 24000 -ac 1 ref_audio_25s.wav
```

---

### VibeVoice Gradio only accessible on localhost

The default launch configuration restricts to `127.0.0.1`. Fix by editing `demo/gradio_demo.py`:

```bash
sed -i 's/# server_port=args.port,/server_port=args.port,/' demo/gradio_demo.py
sed -i 's/server_name="0.0.0.0" if args.share else "127.0.0.1"/server_name="0.0.0.0"/' demo/gradio_demo.py
```

---

### Qwen3-TTS first word cut off

The model sometimes skips the first token of generated speech. Workaround: add a sacrificial word at the start of your text and trim it in post-production:

```bash
# Generate with prefix word
text = "So, this is where your speech really begins..."

# Trim silence/prefix from generated audio
ffmpeg -i output.wav -af silenceremove=start_periods=1:start_threshold=-50dB trimmed.wav
```

---

### Voxtral TTS — first syllable cut off

Voxtral sometimes skips the first syllable of generated speech. Prefix your text with a sacrificial word and trim in post-production:

```bash
# Generate with prefix
./voxtral_tts -d voxtral-tts-model -v neutral_female -o raw.wav "Donc, the technology landscape is evolving rapidly."

# Trim leading silence/prefix
ffmpeg -i raw.wav -af silenceremove=start_periods=1:start_threshold=-50dB trimmed.wav
```

The `voxtral_generate.py` wrapper handles this automatically.

---

### Voxtral TTS — vLLM-Omni not installable on GX10

vLLM-Omni (the official Voxtral serving stack) requires CUDA 12.x and has no prebuilt aarch64 wheel. It cannot be installed on the GX10's CUDA 13.0 + aarch64 environment. Use the pure C binary (`voxtral_tts`) instead. Monitor the vLLM-Omni GitHub for ARM64 support updates.

After the first compilation, the wheel is cached at `~/.cache/pip/wheels/`. Install from cache in subsequent venvs:

```bash
find ~/.cache/pip -name "flash_attn*.whl"
pip install /path/to/cached/flash_attn-*.whl
```
