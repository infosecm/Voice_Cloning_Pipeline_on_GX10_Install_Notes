# Voice Cloning Pipeline — ASUS Ascent GX10

Local voice cloning and synthesis pipeline running entirely on the ASUS Ascent GX10 (NVIDIA GB10 Grace Blackwell Superchip). Supports French and multilingual zero-shot voice cloning using three TTS engines: F5-TTS, CosyVoice 2, and Fish Speech 1.5.

---

## Table of Contents

- [System Specifications](#system-specifications)
- [GX10-Specific Considerations](#gx10-specific-considerations)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Virtual Environment Strategy](#virtual-environment-strategy)
- [Installation — F5-TTS](#installation--f5-tts)
- [Installation — CosyVoice 2](#installation--cosyvoice-2)
- [Installation — Fish Speech 1.5](#installation--fish-speech-15)
- [Scripts Usage](#scripts-usage)
- [Audio Reference Files](#audio-reference-files)
- [Model Comparison](#model-comparison)
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
| Storage | ~1.8 TB NVMe |

---

## GX10-Specific Considerations

The GX10 has several important differences from a standard x86 Linux workstation that affect installation significantly.

**Unified memory architecture.** The GPU does not have dedicated VRAM — it shares the full 128 GB with the CPU. `nvidia-smi` reports `N/A` for memory usage, which is normal. All models load directly into unified memory.

**ARM aarch64.** Many Python packages do not provide prebuilt `aarch64` wheels. Several packages must be compiled from source or pinned to specific versions that provide aarch64 wheels. This is the primary source of installation friction throughout this guide.

**CUDA 13.0 + PyTorch.** The GX10 ships with CUDA 13.0. As of mid-2025, there are no official PyTorch wheels for CUDA 13 on aarch64 published on PyPI or `download.pytorch.org`. The CUDA-enabled PyTorch (`torch 2.11.0+cu130`) is **pre-installed by NVIDIA** as part of the DGX/Spark system stack during the initial machine setup. It lives in whatever Python environment was set up at install time (in this guide, that is `nvidia_torch_venv`). Do not attempt to install it manually via pip — no public wheel exists.

**Torch version pinning.** Several packages (CosyVoice, Fish Speech) attempt to install their own version of PyTorch during setup, which silently overwrites the CUDA-enabled version with a CPU-only build from PyPI. The fix is to always restore the `.pth` pointer to the NVIDIA-provided torch after installing any package that depends on torch.

**ROCm not available.** The machine uses NVIDIA drivers (580.x). Ignore any ROCm-related instructions.

---

## Repository Structure

```
~/voice_clone/
├── fish_speech_clone.sh        # Fish Speech 1.5 pipeline script
├── cosyvoice_clone.sh          # CosyVoice 2 pipeline script
├── start_f5tts_gradio.sh       # F5-TTS Gradio web UI launcher
├── fish-speech/                # Fish Speech source (git clone, tag v1.5.1)
├── CosyVoice/                  # CosyVoice source (git clone)
└── audio_samples/              # Reference audio files and generated outputs
    ├── ref_25s.wav             # Primary audio reference (25s, your voice)
    └── output_*.wav            # Generated audio files (timestamped)

~/f5tts/                        # Dedicated venv for F5-TTS (gradio 6.x)
~/nvidia_torch_venv/            # NVIDIA-provided torch 2.11+cu130 (path varies per system)
```

---

## Prerequisites

### System packages

```bash
sudo apt-get install -y \
    espeak-ng \
    ffmpeg \
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

### PyTorch with CUDA 13 — aarch64

> **Important:** `torch 2.11.0+cu130` for aarch64 is **not available on PyPI**. It is pre-installed by NVIDIA on the GX10 as part of the DGX system stack. It lives in the Python environment that was configured during initial machine setup.
>
> All virtual environments in this guide inherit torch from that environment via a `.pth` pointer file. Do not attempt to install torch via pip in any of these venvs — it will replace the CUDA version with a CPU-only build.

Verify that the NVIDIA torch is available on your system before proceeding:

```bash
# Locate the NVIDIA-provided torch
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected output: 2.11.0+cu130 True
# If this fails, locate the venv that contains it:
find ~ -name "torch" -path "*/site-packages/torch" -type d 2>/dev/null | head -5
```

Note the path to the `site-packages` directory containing torch — you will need it in the next section. In this guide, that path is:
```
<YOUR_NVIDIA_TORCH_SITE_PACKAGES>
```
Replace it with your actual path throughout this guide.

---

## Virtual Environment Strategy

Three venvs are used to avoid dependency conflicts, particularly around `gradio` and `protobuf` versions:

| Venv | Path | Purpose | Gradio version |
|------|------|---------|----------------|
| `voice_clone` | `~/voice_clone/` | CosyVoice 2 + Fish Speech | 5.4.0 |
| `f5tts` | `~/f5tts/` | F5-TTS only | 6.10.0 |
| NVIDIA base | `~/nvidia_torch_venv/` | Source of torch 2.11+cu130 | — |

All venvs inherit torch from the NVIDIA base environment via a `.pth` file. F5-TTS requires its own venv because it needs gradio 6.x while CosyVoice requires gradio 5.x — they cannot coexist in the same venv.

### Create the main voice_clone venv

```bash
python3 -m venv ~/voice_clone

# Point to the NVIDIA-provided torch (replace path if different on your system)
echo "<YOUR_NVIDIA_TORCH_SITE_PACKAGES>" \
    > ~/voice_clone/lib/python3.12/site-packages/nvidia_torch.pth

source ~/voice_clone/bin/activate

# Verify CUDA is available
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.11.0+cu130 True
```

### Create the f5tts venv

```bash
python3 -m venv ~/f5tts

echo "<YOUR_NVIDIA_TORCH_SITE_PACKAGES>" \
    > ~/f5tts/lib/python3.12/site-packages/nvidia_torch.pth
```

---

## Installation — F5-TTS

F5-TTS is a zero-shot voice cloning model. It works well in English but produces noticeable accent artifacts in French (Scandinavian-sounding). Best used for English synthesis or rapid prototyping.

### Install

```bash
source ~/f5tts/bin/activate

pip install f5-tts
pip install "gradio>=6.0.0,<6.11"

# Fix wandb conflict: the NVIDIA base venv has an old wandb that uses distutils
# (removed in Python 3.12). Force-install a newer version in this venv.
~/f5tts/bin/pip install wandb --upgrade --force-reinstall --no-deps

# Verify
python -c "from f5_tts.api import F5TTS; print('F5-TTS OK')"
```

### Quick test

```bash
python -c "
from f5_tts.api import F5TTS
tts = F5TTS()
audio, sr, _ = tts.infer(
    ref_file='~/voice_clone/lib/python3.12/site-packages/f5_tts/infer/examples/basic/basic_ref_en.wav',
    ref_text='Some call me nature, others call me mother nature.',
    gen_text='Hello, this is a voice cloning test on the NVIDIA GB10.',
    file_wave='/tmp/test_f5tts.wav'
)
print(f'Generated {len(audio)/sr:.1f}s of audio — saved to /tmp/test_f5tts.wav')
"
```

### Launch Gradio UI

```bash
~/voice_clone/start_f5tts_gradio.sh
# Access at http://<GX10_IP>:7861
```

---

## Installation — CosyVoice 2

CosyVoice 2 is a multilingual TTS model from Alibaba. It supports French but with a noticeable English accent. Best suited for Chinese or English synthesis. French results are intelligible but not fully natural.

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
echo "<YOUR_NVIDIA_TORCH_SITE_PACKAGES>" \
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
    'Hello, this is a voice cloning test.',
    'Some call me nature, others call me mother nature.',
    '~/voice_clone/audio_samples/ref_fr_25s.wav',
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

Fish Speech is the **recommended engine for French synthesis**. It produces natural-sounding French without foreign accent artifacts, unlike F5-TTS and CosyVoice 2.

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
echo "<YOUR_NVIDIA_TORCH_SITE_PACKAGES>" \
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
REF_AUDIO="~/voice_clone/audio_samples/ref_fr_25s.wav"
SAMPLES="~/voice_clone/audio_samples"

# Step 1: Encode reference audio to VQ tokens
python fish_speech/models/vqgan/inference.py \
    -i "$REF_AUDIO" \
    --checkpoint-path "$CHECKPOINT/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    -o /tmp/ref_tokens_tmp.wav \
    -d cuda
# Tokens saved as: $SAMPLES/ref_fr_25s.npy

# Step 2: Generate semantic codes from text
python fish_speech/models/text2semantic/inference.py \
    --text "Bonjour, ceci est un test de synthese vocale." \
    --prompt-text "Bonjour ! Aujourd hui, je vais vous parler d un sujet fascinant." \
    --prompt-tokens "$SAMPLES/ref_fr_25s.npy" \
    --checkpoint-path "$CHECKPOINT" \
    --output-dir "$SAMPLES" \
    --device cuda \
    --num-samples 1
# Codes saved as: $SAMPLES/codes_0.npy

# Step 3: Decode codes to audio
python fish_speech/models/vqgan/inference.py \
    -i "$SAMPLES/codes_0.npy" \
    --checkpoint-path "$CHECKPOINT/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    -o /tmp/test_fish_speech.wav \
    -d cuda
```

> **Note on prompt text:** Avoid apostrophes and accented characters when passing the reference transcript as a shell argument — they cause quoting errors. Use simplified ASCII text instead (e.g. `"Aujourd hui"` instead of `"Aujourd'hui"`).

---

## Scripts Usage

Three shell scripts automate the most common tasks. All scripts are located in `~/voice_clone/`.

---

### `fish_speech_clone.sh` — Fish Speech voice cloning

The recommended script for French voice cloning.

```bash
# Basic usage — synthesizes text using the default reference voice
./fish_speech_clone.sh "Texte a synthetiser"

# With a custom reference audio and transcript
./fish_speech_clone.sh "Text" /path/to/ref.wav "Transcript of the reference audio"
```

**Output:** `~/voice_clone/audio_samples/output_YYYYMMDD_HHMMSS.wav`

**Notes:**
- Automatically restores NVIDIA torch 2.11+cu130 before running
- Default reference: `~/voice_clone/audio_samples/ref_fr_25s.wav`
- Avoid apostrophes in the reference transcript ($3) — use simplified ASCII text

---

### `cosyvoice_clone.sh` — CosyVoice 2 voice cloning

For Chinese/English synthesis or comparison testing.

```bash
# Basic usage
./cosyvoice_clone.sh "Text to synthesize"

# With custom reference
./cosyvoice_clone.sh "Text" /path/to/ref.wav "Reference transcript"
```

**Output:** `~/voice_clone/audio_samples/output_cosyvoice_YYYYMMDD_HHMMSS.wav`

**Notes:**
- Reference audio must be 30 seconds or less
- Automatically restores NVIDIA torch 2.11+cu130
- French output has noticeable English accent

---

### `start_f5tts_gradio.sh` — F5-TTS web interface

Launches a Gradio web UI for interactive voice cloning with F5-TTS.

```bash
./start_f5tts_gradio.sh
# Access at http://<GX10_IP>:7861
```

**Notes:**
- Uses the dedicated `f5tts` venv (gradio 6.x)
- Keep running in a `screen` or `tmux` session for persistent access
- Built-in audio player, drag-and-drop reference upload

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
ffmpeg -i input.m4a -t 25 -ar 24000 -ac 1 ~/voice_clone/audio_samples/ref_fr_25s.wav

# Transcribe with Whisper on GPU
source ~/voice_clone/bin/activate
python -c "
import whisper
model = whisper.load_model('medium', device='cuda')
result = model.transcribe('ref_fr_25s.wav', language='fr')
print(result['text'])
"
```

---

## Model Comparison

| Model | French quality | Cloning accuracy | Speed (GPU) | Notes |
|-------|---------------|-----------------|-------------|-------|
| Fish Speech 1.5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~15 tok/s | Best for French |
| CosyVoice 2 | ⭐⭐⭐ | ⭐⭐⭐ | RTF ~7 on CPU* | English accent in French |
| F5-TTS | ⭐⭐ | ⭐⭐⭐ | ~3s / 10s audio | Scandinavian accent in French |

*CosyVoice runs on CPU for the ONNX speaker encoder (no aarch64 GPU wheel). The main model runs on GPU.

**Recommendation:** Use Fish Speech 1.5 for all French synthesis.

---

## Troubleshooting

### CUDA not available after installing a new package

Any pip package that depends on `torch` will replace the NVIDIA torch with a CPU-only build from PyPI. This is the most common issue. Fix:

```bash
source ~/voice_clone/bin/activate
echo "<YOUR_NVIDIA_TORCH_SITE_PACKAGES>" \
    > ~/voice_clone/lib/python3.12/site-packages/nvidia_torch.pth
python -c "import torch; print(torch.cuda.is_available())"  # must print True
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
python3 -m venv ~/f5tts
echo "<YOUR_NVIDIA_TORCH_SITE_PACKAGES>" \
    > ~/f5tts/lib/python3.12/site-packages/nvidia_torch.pth
source ~/f5tts/bin/activate
pip install f5-tts "gradio>=6.0.0,<6.11"
~/f5tts/bin/pip install wandb --upgrade --force-reinstall --no-deps
```

---

### `ModuleNotFoundError: No module named 'distutils'` (wandb)

wandb 0.18 in the NVIDIA base venv uses `distutils`, removed in Python 3.12. Fix by installing a newer wandb directly in the target venv, which shadows the old one:

```bash
~/f5tts/bin/pip install wandb --upgrade --force-reinstall --no-deps
```

---

### CosyVoice `AssertionError: do not support extract speech token for audio longer than 30s`

CosyVoice enforces a hard 30-second limit on reference audio. Trim with ffmpeg:

```bash
ffmpeg -i ref_audio.wav -t 25 -ar 24000 -ac 1 ref_audio_25s.wav
```
