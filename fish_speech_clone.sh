#!/bin/bash
# =============================================================================
# Fish Speech 1.5 — Voice Cloning Pipeline
# NVIDIA GB10 / Ubuntu 24.04 aarch64
#
# USAGE:
#   ./fish_speech_clone.sh "Text to synthesize"
#   ./fish_speech_clone.sh "Text to synthesize" /path/to/ref.wav "Reference transcript"
#
# ARGUMENTS:
#   $1  (required) Text to synthesize
#   $2  (optional) Path to reference audio WAV file
#                  Default: ~/voice_clone/audio_samples/ref_fr_25s.wav
#   $3  (optional) Transcript of the reference audio
#                  Default: opening sentences from the French reference recording
#
# EXAMPLES:
#   ./fish_speech_clone.sh "Bonjour, comment allez-vous ?"
#   ./fish_speech_clone.sh "Hello world" ~/voice_clone/audio_samples/ref_en.wav "Hello, this is my reference."
#
# OUTPUT:
#   ~/voice_clone/audio_samples/output_YYYYMMDD_HHMMSS.wav
#
# NOTES:
#   - Requires the voice_clone venv with Fish Speech 1.5 installed
#   - Forces torch 2.11+cu130 from kohya venv (GPU support on GB10)
#   - Pipeline: encode ref audio -> generate semantic codes -> decode to audio
# =============================================================================

set -e

# --- Configuration ---
# Path to the site-packages directory containing torch 2.11+cu130 (pre-installed by NVIDIA on GX10)
# To find it on your system: find ~ -name "torch" -path "*/site-packages/torch" -type d 2>/dev/null
NVIDIA_TORCH_PATH="$HOME/kohya_ss/.venv_kohya/lib/python3.12/site-packages"

FISH_DIR="$HOME/voice_clone/fish-speech"
CHECKPOINT="$FISH_DIR/checkpoints/fish-speech-1.5"
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
OUTPUT="$SAMPLES_DIR/output_${TIMESTAMP}.wav"

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

# --- Restore torch 2.11+cu130 (required after Fish Speech install overwrites it) ---
echo "$NVIDIA_TORCH_PATH" > "$VENV/lib/python3.12/site-packages/nvidia_torch.pth"

echo "=============================================="
echo " Fish Speech 1.5 - Voice Cloning Pipeline"
echo "=============================================="
echo " Reference audio : $REF_AUDIO"
echo " Output          : $OUTPUT"
echo " Text            : $GEN_TEXT"
echo "=============================================="

cd "$FISH_DIR"

# --- Step 1: Encode reference audio to tokens ---
echo ""
echo "[1/3] Encoding reference audio to tokens..."

python fish_speech/models/vqgan/inference.py \
    -i "$REF_AUDIO" \
    --checkpoint-path "$CHECKPOINT/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    -o "$SAMPLES_DIR/ref_tokens_tmp.wav" \
    -d cuda

# vqgan saves .npy next to the input audio file
REF_TOKENS_NPY="$SAMPLES_DIR/$(basename ${REF_AUDIO%.wav}).npy"
if [ ! -f "$REF_TOKENS_NPY" ]; then
    REF_TOKENS_NPY=$(ls -t "$SAMPLES_DIR"/*.npy 2>/dev/null | head -1)
fi
echo " Tokens: $REF_TOKENS_NPY"

# --- Step 2: Generate semantic codes from text ---
echo ""
echo "[2/3] Generating semantic codes from text..."

python fish_speech/models/text2semantic/inference.py \
    --text "$GEN_TEXT" \
    --prompt-text "$REF_TEXT" \
    --prompt-tokens "$REF_TOKENS_NPY" \
    --checkpoint-path "$CHECKPOINT" \
    --output-dir "$SAMPLES_DIR" \
    --device cuda \
    --num-samples 1

echo " Codes: $SAMPLES_DIR/codes_0.npy"

# --- Step 3: Decode codes to audio ---
echo ""
echo "[3/3] Decoding codes to audio..."

python fish_speech/models/vqgan/inference.py \
    -i "$SAMPLES_DIR/codes_0.npy" \
    --checkpoint-path "$CHECKPOINT/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    -o "$OUTPUT" \
    -d cuda

echo ""
echo "=============================================="
echo " Done! Output saved to:"
echo " $OUTPUT"
echo "=============================================="
