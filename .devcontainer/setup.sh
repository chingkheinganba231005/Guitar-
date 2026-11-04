#!/usr/bin/env bash
set -e
sudo apt-get update && sudo apt-get install -y --no-install-recommends ffmpeg libsndfile1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-audio.txt
pip install --no-deps basic-pitch
pip install -e ".[web]"
python -c "from demucs.pretrained import get_model; get_model('htdemucs_6s')"
[ -d /workspaces/atelier-fingerstyle ] || git clone --depth 1 https://github.com/chingkheinganba231005/atelier-fingerstyle /workspaces/atelier-fingerstyle
