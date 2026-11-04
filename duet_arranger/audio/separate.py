"""Source separation with Demucs (htdemucs_6s: vocals, drums, bass, guitar, piano, other)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")


def load_audio(path: str, sr: int = 44100, mono: bool = False) -> tuple[np.ndarray, int]:
    """Decode any audio file to float32 (channels, samples). Uses ffmpeg when needed."""
    try:
        data, file_sr = sf.read(path, always_2d=True, dtype="float32")
        data = data.T
        if file_sr != sr:
            import librosa

            data = np.stack([librosa.resample(c, orig_sr=file_sr, target_sr=sr) for c in data])
    except Exception:
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "decoded.wav")
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path, "-ar", str(sr), "-ac", "2", wav], check=True)
            data, _ = sf.read(wav, always_2d=True, dtype="float32")
            data = data.T
    if data.shape[0] == 1 and not mono:
        data = np.repeat(data, 2, axis=0)
    if mono:
        data = data.mean(axis=0, keepdims=True)
    return data.astype(np.float32), sr


def separate(path: str, out_dir: str, model_name: str = "htdemucs_6s", device: str | None = None, shifts: int = 1) -> dict[str, str]:
    """Split ``path`` into stems written as WAV files in ``out_dir``. Cached."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {s: os.path.join(out_dir, f"{s}.wav") for s in STEMS}
    if all(os.path.exists(p) for p in paths.values()):
        log.info("Using cached stems in %s", out_dir)
        return paths
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model(model_name)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    audio, sr = load_audio(path, model.samplerate)
    wav = torch.from_numpy(audio)
    if wav.shape[0] != model.audio_channels:
        wav = wav[: model.audio_channels] if wav.shape[0] > model.audio_channels else wav.repeat(model.audio_channels, 1)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    log.info("Separating %s with %s on %s …", os.path.basename(path), model_name, device)
    with torch.no_grad():
        sources = apply_model(model, ((wav - mean) / std)[None], device=device, shifts=shifts, split=True, overlap=0.25, progress=log.isEnabledFor(logging.INFO))[0]
    sources = sources * std + mean
    out = {}
    for name, src in zip(model.sources, sources):
        p = os.path.join(out_dir, f"{name}.wav")
        sf.write(p, src.cpu().numpy().T, sr, subtype="PCM_16")
        out[name] = p
    for s in STEMS:
        out.setdefault(s, paths[s])
    return out
