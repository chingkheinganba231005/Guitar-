/* Stem Studio playback engine (AudioWorklet).

   Plays every track from one read head, so they can never drift apart, at
   any speed without changing the pitch: WSOLA time-stretching (windowed
   overlap-add whose frames are nudged to where the waveform lines up). The
   alignment is searched once on the mix you hear and applied to every track,
   so the tracks stay coherent with each other. At 100% the frames are not
   nudged and the overlap-add reconstructs the recording exactly. Loops jump
   inside the overlap-add, so the jump is a smooth crossfade. */

const N = 2048; // frame
const HS = N / 2; // output hop
const DELTA = 400; // alignment search, samples either way
const WIN = new Float32Array(N);
for (let i = 0; i < N; i++) WIN[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);

class StemPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.tracks = new Map(); // id -> { ch: Float32Array[], gain, level }
    this.len = 0;
    this.playing = false;
    this.speed = 1;
    this.loop = { on: false, a: 0, b: 0 };
    this.acc = [new Float32Array(4 * N), new Float32Array(4 * N)];
    this.avail = 0; // finished samples at the start of acc
    this.hops = []; // [{ at, pos }] nominal source position of each frame start
    this.ideal = 0; // nominal source start of the next frame (samples)
    this.prevStart = -1; // actual start of the previous frame
    this.jumped = false;
    this.fade = 0; // samples left of a fade-out (pause)
    this.pausePos = 0;
    this.blocks = 0;
    this.seq = 0;
    this.ended = false;
    this.natMix = new Float32Array(HS);
    this.regMix = new Float32Array(HS + 2 * DELTA);
    this.port.onmessage = (e) => this.onMessage(e.data);
  }

  onMessage(m) {
    if (m.seq !== undefined) this.seq = m.seq;
    if (m.type === "track") {
      this.tracks.set(m.id, { ch: m.channels, gain: m.gain ?? 0, level: 0 });
      this.len = Math.max(this.len, m.channels[0].length);
    } else if (m.type === "gain") {
      const t = this.tracks.get(m.id);
      if (t) t.gain = m.value;
    } else if (m.type === "play") {
      this.reset(m.pos * sampleRate);
      this.playing = true;
      this.ended = false;
    } else if (m.type === "pause") {
      if (this.playing) {
        this.pausePos = this.nominalAt(0);
        this.fade = 256;
      }
    } else if (m.type === "seek") {
      const pos = m.pos * sampleRate;
      if (this.playing) this.reset(pos);
      else this.pausePos = pos;
      this.report(true);
    } else if (m.type === "speed") {
      this.speed = Math.max(0.2, Math.min(2, m.value));
    } else if (m.type === "loop") {
      this.loop = { on: !!m.on, a: m.a * sampleRate, b: m.b * sampleRate };
      if (this.loop.on && this.playing) {
        const p = this.nominalAt(0);
        if (p < this.loop.a - 1 || p >= this.loop.b) this.reset(this.loop.a);
      }
    } else if (m.type === "clear") {
      this.tracks.clear();
      this.len = 0;
      this.playing = false;
    }
  }

  reset(pos) {
    this.acc[0].fill(0);
    this.acc[1].fill(0);
    this.avail = 0;
    this.hops = [];
    this.ideal = Math.max(0, pos);
    this.prevStart = -1;
    this.fade = 0;
  }

  nominalAt(offset) {
    // source position (samples) of output sample ``offset`` in acc
    let h = null;
    for (const x of this.hops) {
      if (x.at <= offset) h = x;
      else break;
    }
    if (!h) return this.hops.length ? this.hops[0].pos : this.playing ? this.ideal : this.pausePos;
    let p = h.pos + (offset - h.at) * h.speed;
    const L = this.loop;
    // the frame that crosses the loop end is the crossfade into its start
    if (L.on && L.b > L.a && p >= L.b) p = L.a + ((p - L.b) % (L.b - L.a));
    return p;
  }

  mixInto(out, from, count) {
    // the mono mix you hear (for the alignment search)
    out.fill(0, 0, count);
    let wsum = 0;
    for (const t of this.tracks.values()) wsum += t.gain;
    const flat = wsum < 1e-4;
    for (const t of this.tracks.values()) {
      const g = flat ? 1 : t.gain;
      if (g < 1e-4) continue;
      const nch = t.ch.length;
      const c0 = t.ch[0], c1 = t.ch[nch - 1];
      const k = nch > 1 ? 0.5 * g : g;
      for (let i = 0; i < count; i++) {
        const s = from + i;
        if (s < 0 || s >= c0.length) continue;
        out[i] += nch > 1 ? k * (c0[s] + c1[s]) : k * c0[s];
      }
    }
  }

  align(start) {
    const nat = this.prevStart + HS;
    this.mixInto(this.natMix, nat, HS);
    this.mixInto(this.regMix, start - DELTA, HS + 2 * DELTA);
    let best = start, bestScore = -Infinity;
    const nm = this.natMix, rm = this.regMix;
    for (let d = 0; d <= 2 * DELTA; d += 2) {
      let dot = 0, en = 1e-9;
      for (let n = 0; n < HS; n += 4) {
        const v = rm[d + n];
        dot += nm[n] * v;
        en += v * v;
      }
      const score = dot / Math.sqrt(en);
      if (score > bestScore) { bestScore = score; best = start - DELTA + d; }
    }
    return best;
  }

  synthFrame() {
    const loop = this.loop;
    if (loop.on && loop.b > loop.a && this.ideal >= loop.b) {
      this.ideal = loop.a + ((this.ideal - loop.b) % Math.max(1, loop.b - loop.a));
      this.jumped = true;
    }
    if (!loop.on && this.ideal >= this.len) {
      this.ended = true;
    }
    let start = Math.round(this.ideal);
    if (this.prevStart >= 0) {
      if (this.speed === 1 && !this.jumped) start = this.prevStart + HS; // exact reconstruction
      else start = this.align(start);
    }
    this.jumped = false;
    const o = this.avail;
    const L = this.acc[0], R = this.acc[1];
    for (const t of this.tracks.values()) {
      const c0 = t.ch[0], c1 = t.ch[t.ch.length - 1];
      // activity meter: the track's own level, whatever its fader
      let e = 0;
      for (let n = 0; n < N; n += 16) {
        const s = start + n;
        if (s >= 0 && s < c0.length) e += c0[s] * c0[s];
      }
      t.level = Math.sqrt(e / (N / 16));
      const g = t.gain;
      if (g < 1e-5) continue;
      for (let n = 0; n < N; n++) {
        const s = start + n;
        if (s < 0 || s >= c0.length) continue;
        const w = WIN[n] * g;
        L[o + n] += w * c0[s];
        R[o + n] += w * c1[s];
      }
    }
    this.hops.push({ at: o, pos: this.ideal, speed: this.speed });
    this.prevStart = start;
    this.ideal += HS * this.speed;
    this.avail += HS;
  }

  report(force) {
    if (!force && this.blocks % 4) return;
    const levels = {};
    if (this.blocks % 8 === 0) for (const [id, t] of this.tracks) levels[id] = t.level;
    this.port.postMessage({
      type: "pos",
      frame: currentFrame,
      pos: (this.playing ? this.nominalAt(0) : this.pausePos) / sampleRate,
      speed: this.speed,
      playing: this.playing,
      seq: this.seq,
      levels,
    });
  }

  process(inputs, outputs) {
    const out = outputs[0];
    const B = out[0].length;
    this.blocks++;
    if (!this.playing || this.len === 0) {
      this.report(false);
      return true;
    }
    while (this.avail < B) this.synthFrame();
    const L = this.acc[0], R = this.acc[1];
    for (let i = 0; i < B; i++) {
      let g = 1;
      if (this.fade > 0) g = Math.max(0, (this.fade - i) / 256);
      out[0][i] = L[i] * g;
      if (out[1]) out[1][i] = R[i] * g;
    }
    // shift what is left
    const keep = this.avail - B + N;
    L.copyWithin(0, B, B + keep);
    R.copyWithin(0, B, B + keep);
    L.fill(0, keep, keep + B);
    R.fill(0, keep, keep + B);
    this.avail -= B;
    for (const h of this.hops) h.at -= B;
    while (this.hops.length > 1 && this.hops[1].at <= 0) this.hops.shift();
    if (this.fade > 0) {
      this.fade -= B;
      if (this.fade <= 0) {
        this.fade = 0;
        this.playing = false;
        this.report(true);
        return true;
      }
    }
    if (this.ended && this.nominalAt(0) >= this.len) {
      this.playing = false;
      this.pausePos = this.len;
      this.port.postMessage({ type: "ended" });
      this.report(true);
      return true;
    }
    this.report(false);
    return true;
  }
}

registerProcessor("stem-player", StemPlayer);
