#!/bin/bash
# =============================================================================
# F5-TTS — Gradio Web UI Launcher
# NVIDIA GB10 / Ubuntu 24.04 aarch64
#
# USAGE:
#   ./start_f5tts_gradio.sh
#
# ACCESS:
#   http://<GX10_IP>:7861
#
# NOTES:
#   - Uses the dedicated f5tts venv (requires gradio 6.x, incompatible with voice_clone venv)
#   - Restores NVIDIA torch 2.11+cu130 before launching
#   - Run inside screen or tmux for persistent access
#
# CONFIGURATION:
#   NVIDIA_TORCH_PATH — path to site-packages containing torch 2.11+cu130
#   To find it: find ~ -name "torch" -path "*/site-packages/torch" -type d 2>/dev/null
# =============================================================================

# Path to the site-packages directory containing torch 2.11+cu130 (pre-installed by NVIDIA on GX10)
NVIDIA_TORCH_PATH="$HOME/kohya_ss/.venv_kohya/lib/python3.12/site-packages"

source ~/f5tts/bin/activate
echo "$NVIDIA_TORCH_PATH" > ~/f5tts/lib/python3.12/site-packages/nvidia_torch.pth
python -m f5_tts.infer.infer_gradio -H 0.0.0.0 -p 7861
