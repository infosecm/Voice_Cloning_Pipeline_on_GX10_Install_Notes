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