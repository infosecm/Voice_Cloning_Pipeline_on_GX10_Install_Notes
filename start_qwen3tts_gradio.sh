#!/bin/bash
echo "🎙️ Starting Qwen3-TTS on Gradio..."

echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/qwen3tts/.venv/lib/python3.12/site-packages/nvidia_torch.pth

source ~/voice_clone/qwen3tts/.venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('✅ CUDA OK:', torch.__version__)"

trap 'deactivate; exit' INT TERM

qwen-tts-demo ~/voice_clone/qwen3tts/models/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 7863 --dtype bfloat16
deactivate