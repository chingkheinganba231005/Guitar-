// Runs the Stem Studio playback worklet outside a browser on test signals:
// pitch must survive slowing down, 100% must reproduce the recording, every
// track must follow the same read head, and loops must stay inside the loop.
const fs = require("fs");
const path = require("path");
const assert = require("assert");

const SR = 44100;
let Proc = null;
const g = {
  sampleRate: SR,
  currentFrame: 0,
  AudioWorkletProcessor: class { constructor() { this.port = { postMessage: (m) => this.port.sent.push(m), sent: [] }; } },
  registerProcessor: (name, cls) => { Proc = cls; },
};
const src = fs.readFileSync(path.join(__dirname, "../../stem_studio/static/player-worklet.js"), "utf8");
new Function("sampleRate", "AudioWorkletProcessor", "registerProcessor", "currentFrame", src)(g.sampleRate, g.AudioWorkletProcessor, g.registerProcessor, 0);

function sine(freq, secs, amp = 0.5) {
  const a = new Float32Array(Math.round(secs * SR));
  for (let i = 0; i < a.length; i++) a[i] = amp * Math.sin((2 * Math.PI * freq * i) / SR);
  return a;
}

function render(p, secs) {
  const n = Math.round(secs * SR / 128);
  const L = new Float32Array(n * 128), R = new Float32Array(n * 128);
  for (let k = 0; k < n; k++) {
    const out = [[new Float32Array(128), new Float32Array(128)]];
    p.process([], out);
    L.set(out[0][0], k * 128);
    R.set(out[0][1], k * 128);
  }
  return [L, R];
}

function freq(x, from, to) {
  let z = 0;
  for (let i = from + 1; i < to; i++) if ((x[i - 1] < 0) !== (x[i] < 0)) z++;
  return (z / 2) / ((to - from) / SR);
}

function rms(x, from, to) {
  let s = 0;
  for (let i = from; i < to; i++) s += x[i] * x[i];
  return Math.sqrt(s / (to - from));
}

function player(tracks, speed = 1) {
  const p = new Proc();
  for (const [id, ch, gain] of tracks) p.onMessage({ type: "track", id, channels: ch, gain });
  p.onMessage({ type: "speed", value: speed });
  p.onMessage({ type: "play", pos: 0 });
  return p;
}

// 1. half speed keeps the pitch and takes twice as long
{
  const p = player([["piano", [sine(440, 4)], 1]], 0.5);
  const [L] = render(p, 4);
  const f = freq(L, SR, 3 * SR);
  assert(Math.abs(f - 440) < 6, `pitch at 50%: ${f}`);
  const pos = p.nominalAt(0) / SR;
  assert(Math.abs(pos - 2) < 0.06, `position after 4 s at 50%: ${pos}`);
  assert(rms(L, SR, 3 * SR) > 0.3, "level kept");
  console.log("50%: pitch", f.toFixed(1), "Hz, position", pos.toFixed(3), "s");
}
// 2. 75% too
{
  const p = player([["piano", [sine(330, 4)], 1]], 0.75);
  const [L] = render(p, 3);
  const f = freq(L, SR / 2, 2.5 * SR);
  assert(Math.abs(f - 330) < 5, `pitch at 75%: ${f}`);
  console.log("75%: pitch", f.toFixed(1));
}
// 3. 100% reproduces the recording exactly (after the first frame's fade-in)
{
  const x = new Float32Array(3 * SR);
  for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.013) * 0.3 + Math.sin(i * 0.071) * 0.2 + (((i * 7919) % 1000) / 1000 - 0.5) * 0.1;
  const p = player([["mix", [x], 1]], 1);
  const [L] = render(p, 2);
  let err = 0;
  for (let i = 2048; i < 1.9 * SR; i++) err = Math.max(err, Math.abs(L[i] - x[i]));
  assert(err < 1e-5, `100% is not exact: ${err}`);
  console.log("100%: max error", err.toExponential(1));
}
// 4. tracks stay locked to one another at 60%: a stem and its copy cancel
{
  const x = sine(220, 4), y = new Float32Array(x.length);
  const z = sine(523, 4, 0.3);
  for (let i = 0; i < x.length; i++) y[i] = -x[i];
  const p = player([["a", [x], 1], ["b", [y], 1], ["c", [z], 1]], 0.6);
  const [L] = render(p, 3);
  // a + b cancel exactly only if both were stretched identically: what is left is c alone
  const f = freq(L, SR / 2, 2.5 * SR);
  assert(Math.abs(f - 523) < 8, `tracks drifted apart: ${f}`);
  console.log("60%: locked tracks, left", f.toFixed(1), "Hz");
}
// 5. a loop stays inside its section, at reduced speed
{
  const p = player([["piano", [sine(440, 10)], 1]], 0.8);
  p.onMessage({ type: "loop", on: true, a: 2, b: 3 });
  let min = 99, max = 0;
  for (let k = 0; k < 6 * SR / 128; k++) {
    p.process([], [[new Float32Array(128), new Float32Array(128)]]);
    if (k > 20) {
      const pos = p.nominalAt(0) / SR;
      min = Math.min(min, pos);
      max = Math.max(max, pos);
    }
  }
  assert(min >= 1.99 && max <= 3.0, `loop escaped: ${min}..${max}`);
  console.log("loop 2-3 s:", min.toFixed(2), "to", max.toFixed(2));
}
// 6. mute: a track at gain 0 is silent
{
  const p = player([["a", [sine(440, 3)], 0]], 1);
  const [L] = render(p, 1);
  assert(rms(L, 0, L.length) < 1e-6, "muted track sounds");
}
console.log("worklet ok");
