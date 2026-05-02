#!/bin/bash
echo "🎙️ Starting F5-TTS on Gradio..."

echo "~/nvidia_torch_venv/lib/python3.12/site-packages" \
    > ~/voice_clone/f5tts/lib/python3.12/site-packages/nvidia_torch.pth

source ~/voice_clone/f5tts/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('✅ CUDA OK:', torch.__version__)"

trap 'deactivate; exit' INT TERM

python -m f5_tts.infer.infer_gradio -H 0.0.0.0 -p 7861
deactivate