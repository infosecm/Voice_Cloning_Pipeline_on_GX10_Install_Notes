#!/bin/bash
# =============================================================================
# CosyVoice 2 — Voice Cloning Pipeline
# NVIDIA GB10 / Ubuntu 24.04 aarch64
#
# USAGE:
#   ./cosyvoice_clone.sh "Text to synthesize"
#   ./cosyvoice_clone.sh "Text to synthesize" /path/to/ref.wav "Reference transcript"
#
# ARGUMENTS:
#   $1  (required) Text to synthesize
#   $2  (optional) Path to reference audio WAV file (max 30 seconds)
#                  Default: ~/voice_clone/audio_samples/ref_en_25s.wav
#   $3  (optional) Transcript of the reference audio
#                  Default: opening sentences from the English reference recording
#
# EXAMPLES:
#   ./cosyvoice_clone.sh "Hello, how are you doing ?"
#   ./cosyvoice_clone.sh "Hello world" ~/voice_clone/audio_samples/ref_en.wav "Hello, this is my reference."
#
# OUTPUT:
#   ~/voice_clone/audio_samples/output_cosyvoice_YYYYMMDD_HHMMSS.wav
#
# NOTES:
#   - Requires the voice_clone venv with CosyVoice 2 installed
#   - Forces torch 2.11+cu130 from nvidia_torch_venv (GPU support on GB10)
#   - Reference audio must be <= 30 seconds (CosyVoice limitation)
# =============================================================================

set -e

# --- Configuration ---
COSYVOICE_DIR="$HOME/voice_clone/CosyVoice"
MODEL_DIR="$COSYVOICE_DIR/pretrained_models/CosyVoice2-0.5B"
SAMPLES_DIR="$HOME/voice_clone/audio_samples"
VENV="$HOME/voice_clone"

# --- Default reference text ---
# Note: apostrophes removed to avoid bash quoting issues.
# To use accented/apostrophe text, pass it explicitly as $3.
DEFAULT_REF_TEXT="Hello! Today I will talk to you about a fascinating subject, artificial intelligence and its impact on our daily lives. Over the past few years, these technologies have transformed the way we work, communicate and even create."

# --- Arguments ---
GEN_TEXT="${1}"
REF_AUDIO="${2:-$SAMPLES_DIR/ref_en_25s.wav}"
REF_TEXT="${3:-$DEFAULT_REF_TEXT}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="$SAMPLES_DIR/output_cosyvoice_${TIMESTAMP}.wav"

# --- Validate input ---
if [ -z "$GEN_TEXT" ]; then
    echo "Usage: $0 \"Text to synthesize\" [ref_audio.wav] [ref_text]"
    echo ""
    echo "Examples:"
    echo "  $0 \"Hello, how are you doing ?\""
    echo "  $0 \"Hello world\" /path/to/ref.wav \"Hello, this is my reference.\""
    exit 1
fi

if [ ! -f "$REF_AUDIO" ]; then
    echo "Error: Reference audio not found: $REF_AUDIO"
    exit 1
fi

# --- Activate venv ---
source "$VENV/bin/activate"

# --- Restore torch 2.11+cu130 (required after CosyVoice install overwrites it) ---
echo "$HOME/nvidia_torch_venv/lib/python3.12/site-packages" > "$VENV/lib/python3.12/site-packages/nvidia_torch.pth"

echo "=============================================="
echo " CosyVoice 2 - Voice Cloning Pipeline"
echo "=============================================="
echo " Reference audio : $REF_AUDIO"
echo " Output          : $OUTPUT"
echo " Text            : $GEN_TEXT"
echo "=============================================="

python - <<EOF
import torch, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.insert(0, '$COSYVOICE_DIR')
sys.path.insert(0, '$COSYVOICE_DIR/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
import torchaudio

print("Loading model...")
cosyvoice = CosyVoice2('$MODEL_DIR', load_jit=False, load_trt=False)
print("Device:", next(cosyvoice.model.flow.parameters()).device)

output = cosyvoice.inference_zero_shot(
    """$GEN_TEXT""",
    """$REF_TEXT""",
    '$REF_AUDIO',
    stream=False
)

for i, result in enumerate(output):
    torchaudio.save('$OUTPUT', result['tts_speech'], cosyvoice.sample_rate)
    print(f"Done! Output saved to: $OUTPUT")
    break
EOF

echo "=============================================="
echo " Done! Output saved to:"
echo " $OUTPUT"
echo "=============================================="
