#!/usr/bin/env bash
# Bring a Codespace up to date and restart the web app:
#   bash .devcontainer/update.sh
set -e
cd /workspaces/Guitar-
git pull --ff-only
if [ -d /workspaces/atelier-fingerstyle/.git ]; then
  git -C /workspaces/atelier-fingerstyle pull --ff-only
else
  git clone --depth 1 https://github.com/chingkheinganba231005/atelier-fingerstyle /workspaces/atelier-fingerstyle
fi
pip install -q -e ".[web]"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
pkill -f "duet-arranger serve" || true
pkill -f "stem-studio serve" || true
sleep 1
nohup duet-arranger serve --host 0.0.0.0 --port 8000 --jobs /workspaces/duet-jobs > /tmp/duet.log 2>&1 &
nohup stem-studio serve --host 0.0.0.0 --port 8001 --jobs /workspaces/studio-jobs > /tmp/studio.log 2>&1 &
sleep 4
echo "Duet Arranger $(python -c 'import duet_arranger; print(duet_arranger.__version__)') is running on port 8000."
echo "Stem Studio $(python -c 'import stem_studio; print(stem_studio.__version__)') is running on port 8001."
echo "Atelier: $(git -C /workspaces/atelier-fingerstyle log -1 --format='%h %s')"
