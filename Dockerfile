# Duet Arranger web app with audio input (Demucs + Basic Pitch) and Atelier.
#   docker build -t duet-arranger .
#   docker run -p 8000:8000 -v duet-jobs:/data duet-arranger
# then open http://localhost:8000. Set DUET_ACCESS_TOKEN to require a token
# when the server is reachable by others.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md requirements-audio.txt ./
COPY duet_arranger ./duet_arranger

# CPU PyTorch keeps the image small; use a CUDA base image and wheels for a GPU.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-audio.txt \
 && pip install --no-cache-dir --no-deps basic-pitch \
 && pip install --no-cache-dir ".[web]"

# Atelier, served by the app at /atelier/ so the duet opens in the same page.
ARG ATELIER_REPO=https://github.com/chingkheinganba231005/atelier-fingerstyle
RUN git clone --depth 1 "$ATELIER_REPO" /app/atelier && rm -rf /app/atelier/.git

# Bake the separation model into the image (about 80 MB) so the first song is fast.
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs_6s')"

ENV ATELIER_DIR=/app/atelier \
    DUET_JOBS_DIR=/data \
    PORT=8000
VOLUME ["/data"]
EXPOSE 8000
CMD ["sh", "-c", "duet-arranger serve --host 0.0.0.0 --port ${PORT} --jobs ${DUET_JOBS_DIR}"]
