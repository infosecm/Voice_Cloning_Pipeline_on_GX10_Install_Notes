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
#                  Default: ~/voice_clone/audio_samples/ref_fr_25s.wav
#   $3  (optional) Transcript of the reference audio
#                  Default: opening sentences from the French reference recording
#
# EXAMPLES:
#   ./cosyvoice_clone.sh "Bonjour, comment allez-vous ?"
#   ./cosyvoice_clone.sh "Hello world" ~/voice_clone/audio_samples/ref_en.wav "Hello, this is my reference."
#
# OUTPUT:
#   ~/voice_clone/audio_samples/output_cosyvoice_YYYYMMDD_HHMMSS.wav
#
# NOTES:
#   - Requires the voice_clone venv with CosyVoice 2 installed
#   - Forces torch 2.11+cu130 from kohya venv (GPU support on GB10)
#   - Reference audio must be <= 30 seconds (CosyVoice limitation)
#   - Best results with French reference audio for French synthesis
# =============================================================================

set -e

# --- Configuration ---
# Path to the site-packages directory containing torch 2.11+cu130 (pre-installed by NVIDIA on GX10)
# To find it on your system: find ~ -name "torch" -path "*/site-packages/torch" -type d 2>/dev/null
NVIDIA_TORCH_PATH="$HOME/kohya_ss/.venv_kohya/lib/python3.12/site-packages"

COSYVOICE_DIR="$HOME/voice_clone/CosyVoice"
MODEL_DIR="$COSYVOICE_DIR/pretrained_models/CosyVoice2-0.5B"
SAMPLES_DIR="$HOME/voice_clone/audio_samples"
VENV="$HOME/voice_clone"

# --- Default reference text ---
# Note: apostrophes removed to avoid bash quoting issues.
# To use accented/apostrophe text, pass it explicitly as $3.
DEFAULT_REF_TEXT="Bonjour ! Aujourd hui, je vais vous parler d un sujet fascinant, l intelligence artificielle et son impact sur notre quotidien. Depuis quelques annees, ces technologies ont transforme notre facon de travailler, de communiquer et meme de creer."

# --- Arguments ---
GEN_TEXT="${1}"
REF_AUDIO="${2:-$SAMPLES_DIR/ref_fr_25s.wav}"
REF_TEXT="${3:-$DEFAULT_REF_TEXT}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="$SAMPLES_DIR/output_cosyvoice_${TIMESTAMP}.wav"

# --- Validate input ---
if [ -z "$GEN_TEXT" ]; then
    echo "Usage: $0 \"Text to synthesize\" [ref_audio.wav] [ref_text]"
    echo ""
    echo "Examples:"
    echo "  $0 \"Bonjour, comment allez-vous ?\""
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
echo "$NVIDIA_TORCH_PATH" > "$VENV/lib/python3.12/site-packages/nvidia_torch.pth"

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
