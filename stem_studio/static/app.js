"use strict";
/* Stem Studio — upload, follow the analysis, then play every instrument in
   sync (Web Audio, sample-accurate) and watch what each one plays. */

const $ = (s, el = document) => el.querySelector(s);
const COLORS = { vocals: "#ff9aa2", piano: "#f4b860", guitar: "#6fd48b", other: "#b39bff", bass: "#7cc4ff", drums: "#c9ced6" };
const SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
const BLACK = new Set([1, 3, 6, 8, 10]);
const LANE_H = { vocals: 96, piano: 150, other: 120, bass: 84, guitar: 64, drums: 64 };

const state = {
  info: null,
  token: "",
  jobs: [],
  current: null, // job id shown
  data: null, // analysis
  tracks: [], // [{id,label,show,notes,chords,hits,silent,vol,mute,solo,lane,...}]
  mode: "original",
  zoom: 8,
  master: 1,
  pollTimer: null,
};

// ------------------------------------------------------------------ api

function tokenFromUrl() {
  const q = new URLSearchParams(location.search).get("token");
  if (q) {
    try { localStorage.setItem("studio-token", q); } catch (e) {}
    return q;
  }
  try { return localStorage.getItem("studio-token") || ""; } catch (e) { return ""; }
}

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (state.token) headers["x-access-token"] = state.token;
  const res = await fetch(path, Object.assign({}, opts, { headers }));
  if (res.status === 401) {
    const t = prompt("This Stem Studio needs its access token:");
    if (t) {
      state.token = t.trim();
      try { localStorage.setItem("studio-token", state.token); } catch (e) {}
      return api(path, opts);
    }
  }
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res;
}

function withToken(url) {
  return state.token ? url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(state.token) : url;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove("show"), 2600);
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function fmt(t, tenths = false) {
  t = Math.max(0, t || 0);
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return tenths ? `${m}:${s.toFixed(1).padStart(4, "0")}` : `${m}:${String(Math.floor(s)).padStart(2, "0")}`;
}

function noteName(p) {
  const names = state.data && state.data.flats ? FLAT : SHARP;
  return names[((p % 12) + 12) % 12] + (Math.floor(p / 12) - 1);
}

// first index i with arr[i] > x (arr sorted), using key(i)
function upper(arr, x, key = (v) => v) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (key(arr[mid]) <= x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// ------------------------------------------------------------------ audio engine

class Engine {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.buffers = {};
    this.nodes = {};
    this.sources = [];
    this.playing = false;
    this.offset = 0;
    this.started = 0;
    this.duration = 0;
    this.loadId = 0;
  }

  ensure() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AC();
      this.master = this.ctx.createGain();
      this.master.connect(this.ctx.destination);
    }
    return this.ctx;
  }

  unload() {
    this.stopSources();
    this.playing = false;
    this.offset = 0;
    for (const n of Object.values(this.nodes)) { try { n.gain.disconnect(); n.analyser.disconnect(); } catch (e) {} }
    this.buffers = {};
    this.nodes = {};
    this.loadId++;
  }

  async fetchBytes(url, onBytes) {
    const res = await fetch(withToken(url));
    if (!res.ok) throw new Error(`Could not load ${url} (${res.status})`);
    const total = +res.headers.get("content-length") || 0;
    if (!res.body || !res.body.getReader) {
      const b = await res.arrayBuffer();
      onBytes(b.byteLength, b.byteLength);
      return b;
    }
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onBytes(got, total);
    }
    const out = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return out.buffer;
  }

  decode(bytes) {
    const ctx = this.ensure();
    return new Promise((resolve, reject) => {
      const p = ctx.decodeAudioData(bytes, resolve, reject);
      if (p && p.then) p.then(resolve, reject);
    });
  }

  // items: [{id, url}]
  async load(items, onProgress) {
    this.unload();
    const myLoad = this.loadId;
    const ctx = this.ensure();
    const sizes = items.map(() => [0, 1]);
    const report = (msg) => {
      const got = sizes.reduce((a, s) => a + s[0], 0);
      const tot = sizes.reduce((a, s) => a + Math.max(s[0], s[1]), 0);
      onProgress(tot ? got / tot : 0, msg);
    };
    // download three at a time, decode one at a time (keeps memory peaks low)
    const bytes = new Array(items.length);
    let next = 0;
    const worker = async () => {
      while (next < items.length) {
        const i = next++;
        bytes[i] = await this.fetchBytes(items[i].url, (g, t) => { sizes[i] = [g, t || g]; report("Downloading the tracks…"); });
      }
    };
    await Promise.all([worker(), worker(), worker()]);
    for (let i = 0; i < items.length; i++) {
      if (myLoad !== this.loadId) return false;
      onProgress(0.9 + 0.1 * (i / items.length), `Decoding ${items[i].label || items[i].id}…`);
      const buf = await this.decode(bytes[i]);
      bytes[i] = null;
      this.buffers[items[i].id] = buf;
      const gain = ctx.createGain();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      analyser.connect(gain);
      gain.connect(this.master);
      this.nodes[items[i].id] = { gain, analyser, data: new Float32Array(analyser.fftSize) };
      this.duration = Math.max(this.duration, buf.duration);
    }
    return myLoad === this.loadId;
  }

  position() {
    if (!this.playing) return this.offset;
    return Math.min(this.duration, this.offset + Math.max(0, this.ctx.currentTime - this.started));
  }

  // what you hear now (the output buffer lags the clock)
  heard() {
    if (!this.playing || !this.ctx) return this.offset;
    const lat = (this.ctx.outputLatency || 0) + (this.ctx.baseLatency || 0);
    return Math.max(this.offset, this.position() - lat);
  }

  stopSources() {
    for (const s of this.sources) { try { s.stop(); s.disconnect(); } catch (e) {} }
    this.sources = [];
  }

  play() {
    const ctx = this.ensure();
    if (ctx.state === "suspended") ctx.resume();
    if (this.offset >= this.duration - 0.05) this.offset = 0;
    this.stopSources();
    const when = ctx.currentTime + 0.06;
    for (const [id, buf] of Object.entries(this.buffers)) {
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.nodes[id].analyser);
      src.start(when, Math.min(this.offset, buf.duration));
      this.sources.push(src);
    }
    this.started = when;
    this.playing = true;
  }

  pause() {
    if (!this.playing) return;
    this.offset = this.position();
    this.stopSources();
    this.playing = false;
  }

  seek(t) {
    t = Math.max(0, Math.min(this.duration || t, t));
    const was = this.playing;
    if (was) { this.stopSources(); this.playing = false; }
    this.offset = t;
    if (was) this.play();
  }

  setGain(id, v) {
    const n = this.nodes[id];
    if (!n) return;
    n.gain.gain.setTargetAtTime(v, this.ctx.currentTime, 0.015);
  }

  setMaster(v) {
    if (this.master) this.master.gain.setTargetAtTime(v, this.ctx.currentTime, 0.015);
  }

  level(id) {
    const n = this.nodes[id];
    if (!n || !this.playing) return 0;
    n.analyser.getFloatTimeDomainData(n.data);
    let s = 0;
    for (let i = 0; i < n.data.length; i++) s += n.data[i] * n.data[i];
    return Math.sqrt(s / n.data.length);
  }
}

const engine = new Engine();

// ------------------------------------------------------------------ songs list, upload, progress

async function loadInfo() {
  state.info = await (await api("/api/info")).json();
  $("#version").textContent = "v" + state.info.version;
  if (!state.info.audio.available) {
    const p = $("#audio-status");
    p.hidden = false;
    p.textContent = "Separation needs: " + state.info.audio.missing.join(", ");
  }
  $("#file").accept = state.info.accept.join(",") + ",audio/*,video/*";
}

async function refreshJobs() {
  try {
    state.jobs = await (await api("/api/jobs")).json();
  } catch (e) {
    return;
  }
  renderSongs();
  const busy = state.jobs.some((j) => j.state === "queued" || j.state === "running");
  clearTimeout(state.pollTimer);
  if (busy) state.pollTimer = setTimeout(refreshJobs, 2000);
  const cur = state.jobs.find((j) => j.id === state.current);
  if (cur && !$("#working").hidden) renderWorking(cur);
}

function renderSongs() {
  const ul = $("#songs");
  ul.innerHTML = "";
  $("#songs-empty").hidden = state.jobs.length > 0;
  for (const j of state.jobs) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    if (j.id === state.current) b.classList.add("active");
    let sub;
    if (j.state === "done") sub = [fmt(j.duration), j.summary.key, j.summary.meter, j.summary.tempo].filter(Boolean).join(" · ");
    else if (j.state === "failed") sub = "Failed";
    else if (j.state === "cancelled") sub = "Cancelled";
    else sub = `${Math.round(j.progress * 100)}% · ${j.message}`;
    b.innerHTML = `<strong>${esc(j.title)}</strong><small>${esc(sub)}</small>` + (j.state === "running" || j.state === "queued" ? `<div class="mini"><span style="width:${Math.round(j.progress * 100)}%"></span></div>` : "");
    b.onclick = () => openJob(j.id);
    li.appendChild(b);
    ul.appendChild(li);
  }
}

function show(view) {
  for (const id of ["welcome", "working", "player"]) $("#" + id).hidden = id !== view;
}

function renderWorking(job) {
  $("#work-title").textContent = job.title;
  $("#work-fill").style.width = Math.round(job.progress * 100) + "%";
  $("#work-msg").textContent = job.message;
  const el = job.started ? (Date.now() / 1000 - job.started) : 0;
  $("#work-time").textContent = job.state === "running" ? `${Math.round(job.progress * 100)}% · ${fmt(el)}` : job.state;
  const ol = $("#work-stages");
  const ids = job.stages.map((s) => s.id);
  const at = ids.indexOf(job.stage);
  ol.innerHTML = job.stages.map((s, i) => `<li class="${job.state === "done" || i < at ? "done" : i === at ? "now" : ""}">${i < at || job.state === "done" ? "✓ " : ""}${esc(s.label)}</li>`).join("");
  $("#cancel").hidden = !(job.state === "running" || job.state === "queued");
  const err = $("#work-error");
  err.hidden = !job.error;
  err.textContent = job.error || "";
  if (job.state === "done") openJob(job.id);
}

function setupUpload() {
  const drop = $("#drop");
  const file = $("#file");
  const pick = (f) => {
    if (!f) return;
    file._file = f;
    $("#drop-title").textContent = f.name;
    $("#drop-sub").textContent = `${(f.size / 1048576).toFixed(1)} MB · ready`;
    $("#submit").disabled = false;
  };
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") file.click(); });
  file.addEventListener("change", () => pick(file.files[0]));
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
  drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));
  $("#upload").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = file._file;
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f, f.name);
    fd.append("title", $("#title").value.trim());
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    if (state.token) xhr.setRequestHeader("x-access-token", state.token);
    const bar = $("#upload-bar");
    bar.hidden = false;
    $("#submit").disabled = true;
    $("#form-error").hidden = true;
    xhr.upload.onprogress = (ev) => { if (ev.lengthComputable) $("span", bar).style.width = Math.round((ev.loaded / ev.total) * 100) + "%"; };
    xhr.onload = () => {
      bar.hidden = true;
      $("#submit").disabled = false;
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (err) {}
      if (xhr.status >= 300) {
        $("#form-error").hidden = false;
        $("#form-error").textContent = body.detail || `Upload failed (${xhr.status})`;
        return;
      }
      file._file = null;
      file.value = "";
      $("#title").value = "";
      $("#drop-title").textContent = "Drop a song or a video";
      $("#drop-sub").textContent = "MP3, WAV, M4A, FLAC · MP4, MOV, MKV, WEBM";
      $("#submit").disabled = true;
      refreshJobs().then(() => openJob(body.id));
    };
    xhr.onerror = () => {
      bar.hidden = true;
      $("#submit").disabled = false;
      $("#form-error").hidden = false;
      $("#form-error").textContent = "Upload failed — is the app still running?";
    };
    xhr.send(fd);
  });
  $("#cancel").onclick = async () => {
    if (state.current) await api(`/api/jobs/${state.current}/cancel`, { method: "POST" });
    refreshJobs();
  };
  $("#delete").onclick = async () => {
    const j = state.jobs.find((x) => x.id === state.current);
    if (!j || !confirm(`Delete “${j.title}” and its tracks?`)) return;
    engine.unload();
    await api(`/api/jobs/${j.id}`, { method: "DELETE" });
    state.current = null;
    history.replaceState(null, "", location.pathname + location.search);
    show("welcome");
    refreshJobs();
  };
}

async function openJob(id) {
  const job = state.jobs.find((j) => j.id === id);
  if (!job) return;
  const same = state.current === id && state.data && !$("#player").hidden;
  state.current = id;
  history.replaceState(null, "", location.pathname + location.search + "#song=" + id);
  renderSongs();
  if (job.state !== "done") {
    if (state.data) { engine.unload(); state.data = null; }
    show("working");
    renderWorking(job);
    return;
  }
  if (same) return;
  show("player");
  await openPlayer(job);
}

// ------------------------------------------------------------------ player

async function openPlayer(job) {
  engine.unload();
  state.data = null;
  $("#song-title").textContent = job.title;
  $("#lanes").innerHTML = "";
  $("#now-grid").innerHTML = "";
  const data = await (await api(`/api/jobs/${job.id}/analysis`)).json();
  if (state.current !== job.id) return;
  state.data = data;
  state.mode = "original";
  state.tracks = data.tracks.map((t) => Object.assign({}, t, { vol: 1, mute: false, solo: false, maxDur: t.notes.reduce((m, n) => Math.max(m, n[1] - n[0]), 0) }));
  $("#song-facts").innerHTML = [data.key, data.meter, data.tempo_label, `${data.bars.length} bars`, fmt(data.duration)].map((x) => `<span>${esc(x)}</span>`).join("");
  $("#dur").textContent = "/ " + fmt(data.duration);
  buildLanes();
  buildNow();
  setMode("original", true);
  const items = [{ id: "original", label: "the original", url: `/api/jobs/${job.id}/audio/${data.original.split("/").pop()}` }].concat(
    state.tracks.map((t) => ({ id: t.id, label: t.label.toLowerCase(), url: `/api/jobs/${job.id}/audio/${t.audio.split("/").pop()}` }))
  );
  const box = $("#loading");
  box.hidden = false;
  $("#play").disabled = true;
  try {
    const ok = await engine.load(items, (f, msg) => {
      $("#load-fill").style.width = Math.round(f * 100) + "%";
      $("#load-msg").textContent = msg;
    });
    if (!ok) return;
  } catch (e) {
    $("#load-msg").textContent = "Could not load the tracks: " + e.message;
    return;
  }
  box.hidden = true;
  $("#play").disabled = false;
  engine.duration = data.duration;
  state.wave = wavePeaks(engine.buffers.original, 800);
  applyMix();
  engine.setMaster(state.master);
}

function wavePeaks(buf, n) {
  if (!buf) return null;
  const ch = buf.getChannelData(0);
  const step = Math.max(1, Math.floor(ch.length / n));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let m = 0;
    const a = i * step, b = Math.min(ch.length, a + step);
    for (let k = a; k < b; k += 16) { const v = Math.abs(ch[k]); if (v > m) m = v; }
    out[i] = m;
  }
  let mx = 0;
  for (const v of out) mx = Math.max(mx, v);
  if (mx > 0) for (let i = 0; i < n; i++) out[i] /= mx;
  return out;
}

function setMode(mode, quiet) {
  state.mode = mode;
  $("#mode-original").setAttribute("aria-checked", mode === "original");
  $("#mode-stems").setAttribute("aria-checked", mode === "stems");
  applyMix();
  if (!quiet) toast(mode === "original" ? "Listening to the original recording" : "Listening to the separated tracks — use each track's fader");
}

function applyMix() {
  const anySolo = state.tracks.some((t) => t.solo);
  for (const t of state.tracks) {
    const on = anySolo ? t.solo : !t.mute;
    engine.setGain(t.id, state.mode === "stems" && on ? t.vol : 0);
    if (t.lane) t.lane.classList.toggle("dim", state.mode === "stems" && !on);
  }
  engine.setGain("original", state.mode === "original" ? 1 : 0);
}

function touchMix() {
  if (state.mode !== "stems") setMode("stems");
  else applyMix();
}

function buildLanes() {
  const wrap = $("#lanes");
  wrap.innerHTML = "";
  for (const t of state.tracks) {
    const c = COLORS[t.id] || "#aaa";
    const lane = document.createElement("div");
    lane.className = "lane" + (t.silent ? " silent" : "");
    lane.style.setProperty("--c", c);
    const what = t.silent ? "nothing found" : t.show === "chords" ? "chords" : t.show === "hits" ? "hits" : t.show === "notes+chords" ? "notes & chords" : "notes";
    const keysBtn = t.show.startsWith("notes") ? `<button class="k" title="Show a keyboard">⌨</button>` : "";
    lane.innerHTML = `
      <div class="lane-head">
        <div class="lane-title"><i></i>${esc(t.label)}<small>${what}</small></div>
        <div class="lane-controls">
          <button class="m" title="Mute">M</button><button class="s" title="Solo">S</button>
          <input type="range" min="0" max="1.5" step="0.01" value="1" aria-label="${esc(t.label)} volume" />
          ${keysBtn}
          <a href="${withToken(`/api/jobs/${state.current}/audio/${t.audio.split("/").pop()}?download=1`)}" title="Download this track" aria-label="Download ${esc(t.label)}">⬇</a>
        </div>
        <div class="meter"><span></span></div>
        <div class="lane-now">–</div>
      </div>
      <div class="lane-body"><canvas class="roll"></canvas></div>`;
    wrap.appendChild(lane);
    t.lane = lane;
    t.canvas = $("canvas.roll", lane);
    t.canvas.style.height = (LANE_H[t.id] || 90) + "px";
    t.meterEl = $(".meter span", lane);
    t.nowEl = $(".lane-now", lane);
    const mBtn = $("button.m", lane), sBtn = $("button.s", lane);
    mBtn.onclick = () => { t.mute = !t.mute; mBtn.classList.toggle("on", t.mute); touchMix(); };
    sBtn.onclick = () => { t.solo = !t.solo; sBtn.classList.toggle("on", t.solo); touchMix(); };
    $("input", lane).oninput = (e) => { t.vol = +e.target.value; touchMix(); };
    const kb = $("button.k", lane);
    if (kb) kb.onclick = () => toggleKeys(t);
    // pitch range for the roll: where nearly all its notes are (a stray
    // harmonic far above stays on the keyboard, not in the roll), each
    // semitone tall enough to carry the note's name
    if (t.notes.length) {
      const ps = t.notes.map((n) => n[2]).sort((a, b) => a - b);
      let lo = ps[Math.floor(ps.length * 0.03)] - 1, hi = ps[Math.min(ps.length - 1, Math.floor(ps.length * 0.97))] + 1;
      while (hi - lo < 14) { lo--; hi++; }
      t.lo = lo; t.hi = hi;
      const strip = t.show === "notes+chords" ? 18 : 0;
      const rh = t.id === "piano" ? 7 : 9;
      t.canvas.style.height = Math.max(80, Math.min(340, (hi - lo + 1) * rh + strip + 14)) + "px";
    }
    if (t.silent) t.canvas.style.height = "34px";
    attachPointer(t);
    if (t.id === "piano" && !t.silent) toggleKeys(t);
  }
}

function toggleKeys(t) {
  if (t.keys) {
    t.keys.remove();
    t.keys = null;
  } else {
    t.keys = document.createElement("canvas");
    t.keys.className = "keys";
    t.keys.style.height = "58px";
    $(".lane-body", t.lane).appendChild(t.keys);
  }
  const b = $("button.k", t.lane);
  if (b) b.classList.toggle("on", !!t.keys);
}

function buildNow() {
  const g = $("#now-grid");
  g.innerHTML = "";
  for (const t of state.tracks) {
    if (t.silent) continue;
    const d = document.createElement("div");
    d.style.setProperty("--c", COLORS[t.id]);
    d.innerHTML = `<small>${esc(t.label)}</small><b>–</b>`;
    g.appendChild(d);
    t.nowCell = d;
  }
}

// ------------------------------------------------------------------ timing helpers

function barBeat(t) {
  const d = state.data;
  if (!d || !d.bars.length) return null;
  const bi = Math.max(0, upper(d.bars, t) - 1);
  const b0 = d.bars[bi];
  const b1 = bi + 1 < d.bars.length ? d.bars[bi + 1] : Infinity;
  const i0 = upper(d.beats, b0 - 1e-4);
  const iNow = upper(d.beats, t) - 1;
  const i1 = b1 === Infinity ? d.beats.length : upper(d.beats, b1 - 1e-4);
  const beat = t < b0 ? 1 : Math.max(1, iNow - i0 + 2 - 1);
  return { bar: bi + 1, beat: Math.min(beat, Math.max(1, i1 - i0)), of: Math.max(1, i1 - i0) };
}

function segAt(segs, t) {
  const i = upper(segs, t, (s) => s[0]) - 1;
  return i >= 0 && segs[i][1] > t ? i : -1;
}

function sounding(t, tr, slack = 0) {
  const out = [];
  const ns = tr.notes;
  const j = upper(ns, t, (n) => n[0]);
  for (let i = j - 1; i >= 0 && ns[i][0] >= t - tr.maxDur - 0.01; i--) if (ns[i][1] > t - slack) out.push(ns[i]);
  return out.sort((a, b) => a[2] - b[2]);
}

// ------------------------------------------------------------------ drawing

function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, w, h];
}

const PAST = 0.3; // share of the lane showing what just happened

function laneWindow(t) {
  const W = state.zoom;
  const t0 = t - W * PAST;
  return [t0, t0 + W, W];
}

function drawGrid(ctx, w, h, t0, W, top = 0) {
  const d = state.data;
  const x = (s) => ((s - t0) / W) * w;
  ctx.lineWidth = 1;
  const i0 = upper(d.beats, t0) - 1;
  for (let i = Math.max(0, i0); i < d.beats.length && d.beats[i] < t0 + W; i++) {
    ctx.strokeStyle = "#232a33";
    const X = Math.round(x(d.beats[i])) + 0.5;
    ctx.beginPath(); ctx.moveTo(X, top); ctx.lineTo(X, h); ctx.stroke();
  }
  const b0 = upper(d.bars, t0) - 1;
  ctx.font = "10px system-ui, sans-serif";
  for (let i = Math.max(0, b0); i < d.bars.length && d.bars[i] < t0 + W; i++) {
    const X = Math.round(x(d.bars[i])) + 0.5;
    ctx.strokeStyle = "#3a4452";
    ctx.beginPath(); ctx.moveTo(X, top); ctx.lineTo(X, h); ctx.stroke();
    ctx.fillStyle = "#6c7684";
    ctx.fillText(String(i + 1), X + 3, h - 4);
  }
}

function drawChordStrip(ctx, w, y, hh, segs, t0, W, t, color, big) {
  const x = (s) => ((s - t0) / W) * w;
  const i0 = Math.max(0, upper(segs, t0, (s) => s[0]) - 1);
  ctx.font = (big ? "600 15px" : "600 11.5px") + " system-ui, sans-serif";
  ctx.textBaseline = "middle";
  for (let i = i0; i < segs.length && segs[i][0] < t0 + W; i++) {
    const [a, b, name] = segs[i];
    if (b < t0) continue;
    const X0 = Math.max(0, x(a)), X1 = Math.min(w, x(b));
    const now = a <= t && t < b;
    ctx.fillStyle = now ? color + "55" : "#20262e";
    ctx.fillRect(X0 + 1, y + 1, Math.max(1, X1 - X0 - 2), hh - 2);
    ctx.strokeStyle = now ? color : "#323a45";
    ctx.strokeRect(X0 + 1.5, y + 1.5, Math.max(1, X1 - X0 - 3), hh - 3);
    if (X1 - X0 > 18) {
      ctx.fillStyle = now ? "#fff" : "#c8cfd8";
      ctx.save();
      ctx.beginPath(); ctx.rect(X0, y, X1 - X0, hh); ctx.clip();
      ctx.fillText(name, Math.max(X0 + 5, Math.min(X0 + 5, X1 - 40)), y + hh / 2);
      ctx.restore();
    }
  }
  ctx.textBaseline = "alphabetic";
}

function drawLane(tr, t) {
  const cv = tr.canvas;
  const [ctx, w, h] = fitCanvas(cv);
  ctx.fillStyle = "#13171c";
  ctx.fillRect(0, 0, w, h);
  const [t0, , W] = laneWindow(t);
  const color = COLORS[tr.id] || "#aaa";
  const x = (s) => ((s - t0) / W) * w;
  if (tr.show === "chords") {
    drawGrid(ctx, w, h, t0, W);
    drawChordStrip(ctx, w, 6, h - 20, tr.chords, t0, W, t, color, true);
  } else if (tr.show === "hits") {
    drawGrid(ctx, w, h, t0, W);
    const rows = { hat: 0.22, snare: 0.5, kick: 0.78 };
    ctx.font = "10px system-ui"; ctx.fillStyle = "#5d6673";
    for (const [k, r] of Object.entries(rows)) ctx.fillText(k, 4, r * (h - 12) + 3);
    const hs = tr.hits;
    for (let i = Math.max(0, upper(hs, t0, (q) => q[0]) - 1); i < hs.length && hs[i][0] < t0 + W; i++) {
      const [s, kind, str] = hs[i];
      const X = x(s), Y = (rows[kind] || 0.5) * (h - 12);
      const near = Math.abs(s - t) < 0.08;
      ctx.fillStyle = near ? "#fff" : s < t ? "#8a929c" : color;
      ctx.beginPath(); ctx.arc(X, Y, 2.5 + 3 * Math.min(1, str), 0, Math.PI * 2); ctx.fill();
    }
  } else {
    const strip = tr.show === "notes+chords" ? 18 : 0;
    const top = strip;
    const lo = tr.lo ?? 48, hi = tr.hi ?? 72;
    const rows = hi - lo + 1;
    const rh = (h - top - 12) / rows;
    const y = (p) => top + (hi - p) * rh;
    for (let p = lo; p <= hi; p++) {
      if (BLACK.has(((p % 12) + 12) % 12)) { ctx.fillStyle = "#101318"; ctx.fillRect(0, y(p), w, rh); }
      if (p % 12 === 0 && rh * 12 > 30) {
        ctx.fillStyle = "#4b5461"; ctx.font = "9.5px system-ui";
        ctx.fillText(noteName(p), 3, y(p) + rh - 1);
      }
    }
    drawGrid(ctx, w, h, t0, W, top);
    if (strip) drawChordStrip(ctx, w, 0, strip, tr.chords, t0, W, t, color, false);
    const ns = tr.notes;
    const i0 = Math.max(0, upper(ns, t0 - tr.maxDur, (n) => n[0]) - 1);
    ctx.font = "600 10px system-ui";
    ctx.textBaseline = "middle";
    tr.hitBoxes = [];
    for (let i = i0; i < ns.length && ns[i][0] < t0 + W; i++) {
      const [a, b, p, v] = ns[i];
      if (b < t0 || p < lo || p > hi) continue;
      const X0 = x(a), X1 = Math.max(X0 + 2, x(b));
      const Y = y(p);
      const on = a <= t && t < b;
      ctx.globalAlpha = on ? 1 : a > t ? 0.85 : 0.4;
      ctx.fillStyle = on ? "#ffffff" : color;
      const r = Math.min(3, rh / 2);
      roundRect(ctx, X0, Y + 0.5, X1 - X0, Math.max(2, rh - 1), r);
      ctx.fill();
      if (on) { ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1; }
      if (X1 - X0 > 20 && rh >= 6) {
        ctx.fillStyle = on ? "#111" : "#111c";
        ctx.font = `600 ${Math.min(11, Math.max(7.5, rh))}px system-ui`;
        ctx.fillText(noteName(p), X0 + 3, Y + rh / 2 + 0.5);
      }
      tr.hitBoxes.push([X0, Y, X1, Y + rh, ns[i]]);
    }
    ctx.globalAlpha = 1;
    ctx.textBaseline = "alphabetic";
  }
  // playhead
  const P = Math.round(x(t)) + 0.5;
  ctx.strokeStyle = "#ffffffcc";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(P, 0); ctx.lineTo(P, h); ctx.stroke();
  ctx.lineWidth = 1;
  if (tr.keys) drawKeys(tr, t);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawKeys(tr, t) {
  const [ctx, w, h] = fitCanvas(tr.keys);
  ctx.fillStyle = "#0d1014";
  ctx.fillRect(0, 0, w, h);
  let lo = Math.floor(((tr.lo ?? 48) - 0) / 12) * 12, hi = Math.ceil(((tr.hi ?? 72) + 1) / 12) * 12 - 1;
  lo = Math.max(21, Math.min(lo, 48)); hi = Math.min(108, Math.max(hi, 83));
  const whites = [];
  for (let p = lo; p <= hi; p++) if (!BLACK.has(p % 12)) whites.push(p);
  const kw = w / whites.length;
  const on = new Map(sounding(t, tr).map((n) => [n[2], n]));
  const color = COLORS[tr.id];
  const pos = {};
  whites.forEach((p, i) => { pos[p] = i * kw; });
  ctx.font = "600 10px system-ui";
  ctx.textAlign = "center";
  for (const p of whites) {
    const X = pos[p];
    ctx.fillStyle = on.has(p) ? color : "#e9edf2";
    ctx.fillRect(X + 0.5, 0, kw - 1, h);
    if (on.has(p) && kw > 14) { ctx.fillStyle = "#111"; ctx.fillText(noteName(p), X + kw / 2, h - 7); }
    else if (p % 12 === 0 && kw > 9) { ctx.fillStyle = "#8a929c"; ctx.fillText(noteName(p), X + kw / 2, h - 7); }
  }
  for (let p = lo; p <= hi; p++) {
    if (!BLACK.has(p % 12) || pos[p - 1] === undefined) continue;
    const X = pos[p - 1] + kw * 0.68;
    ctx.fillStyle = on.has(p) ? color : "#1b1f25";
    ctx.fillRect(X, 0, kw * 0.64, h * 0.6);
    if (on.has(p) && kw > 16) { ctx.fillStyle = "#111"; ctx.font = "600 9px system-ui"; ctx.fillText(noteName(p), X + kw * 0.32, h * 0.6 - 6); ctx.font = "600 10px system-ui"; }
  }
  ctx.textAlign = "left";
}

function drawOverview(t) {
  const cv = $("#overview");
  const [ctx, w, h] = fitCanvas(cv);
  const d = state.data;
  ctx.fillStyle = "#13171c";
  ctx.fillRect(0, 0, w, h);
  if (!d) return;
  const D = d.duration || 1;
  const x = (s) => (s / D) * w;
  if (state.wave) {
    ctx.fillStyle = "#2b333e";
    const n = state.wave.length;
    for (let i = 0; i < n; i++) {
      const v = state.wave[i] * (h - 22) * 0.5;
      ctx.fillRect((i / n) * w, 18 + (h - 22) / 2 - v, Math.max(1, w / n), Math.max(1, v * 2));
    }
  }
  // chords along the top
  ctx.font = "600 10px system-ui";
  ctx.textBaseline = "middle";
  let lastName = null;
  for (const [a, b, name] of d.chords) {
    const X0 = x(a), X1 = x(b);
    ctx.fillStyle = name === lastName ? "#262d36" : "#2e3641";
    ctx.fillRect(X0, 0, Math.max(1, X1 - X0 - 1), 16);
    if (X1 - X0 > 16) { ctx.fillStyle = "#c8cfd8"; ctx.fillText(name, X0 + 3, 8); }
    lastName = name;
  }
  ctx.textBaseline = "alphabetic";
  // bar numbers every few bars
  const every = Math.max(1, Math.ceil(d.bars.length / Math.max(1, w / 34)));
  ctx.fillStyle = "#6c7684";
  ctx.font = "9.5px system-ui";
  d.bars.forEach((b, i) => {
    if (i % every) return;
    const X = Math.round(x(b)) + 0.5;
    ctx.fillRect(X, h - 6, 1, 6);
    ctx.fillText(String(i + 1), X + 2, h - 1);
  });
  // window shown in the lanes
  const [t0, t1] = laneWindow(t);
  ctx.fillStyle = "#f4b86022";
  ctx.fillRect(x(t0), 16, x(t1) - x(t0), h - 16);
  ctx.fillStyle = "#f4b860";
  ctx.fillRect(Math.round(x(t)) - 1, 0, 2, h);
}

// ------------------------------------------------------------------ live text

let lastText = 0;
function updateText(t) {
  const now = performance.now();
  if (now - lastText < 90) return;
  lastText = now;
  const d = state.data;
  $("#time").textContent = fmt(t, true);
  const bb = barBeat(t);
  $("#barbeat").textContent = bb ? `Bar ${bb.bar} · ${bb.beat}/${bb.of}` : "";
  const ci = segAt(d.chords, t);
  $("#now-chord").textContent = ci >= 0 ? d.chords[ci][2] : "–";
  const ni = ci >= 0 ? ci + 1 : upper(d.chords, t, (s) => s[0]);
  const nxt = d.chords[ni];
  $("#next-chord").textContent = nxt ? `next ${nxt[2]} in ${Math.max(0, nxt[0] - t).toFixed(1)} s` : "";
  for (const tr of state.tracks) {
    let txt = "–";
    if (tr.show === "chords") {
      const i = segAt(tr.chords, t);
      txt = i >= 0 ? tr.chords[i][2] : "–";
      const n = tr.chords[i >= 0 ? i + 1 : upper(tr.chords, t, (s) => s[0])];
      if (n) txt += `  → ${n[2]}`;
    } else if (tr.show === "hits") {
      const near = tr.hits.filter((h) => h[0] <= t && t - h[0] < 0.15).map((h) => h[1]);
      txt = near.length ? [...new Set(near)].join(" + ") : "–";
    } else {
      const s = sounding(t, tr);
      if (s.length) txt = s.map((n) => noteName(n[2])).join(" ");
      if (tr.show === "notes+chords") {
        const i = segAt(tr.chords, t);
        if (i >= 0) txt = `${tr.chords[i][2]} · ${txt}`;
      }
    }
    if (tr.nowEl.textContent !== txt) tr.nowEl.textContent = txt;
    if (tr.nowCell) {
      const b = $("b", tr.nowCell);
      if (b.textContent !== txt) b.textContent = txt;
      tr.nowCell.classList.toggle("off", txt === "–");
    }
  }
  $("#play").textContent = engine.playing ? "❚❚" : "▶";
  $("#play").setAttribute("aria-label", engine.playing ? "Pause" : "Play");
}

function frame() {
  requestAnimationFrame(frame);
  if (!state.data || $("#player").hidden) return;
  let t = engine.heard();
  if (engine.playing && engine.position() >= engine.duration - 0.01) {
    engine.pause();
    engine.offset = engine.duration;
  }
  for (const tr of state.tracks) {
    drawLane(tr, t);
    const lv = Math.min(1, engine.level(tr.id) * 4);
    tr.meterEl.style.width = Math.round(lv * 100) + "%";
  }
  drawOverview(t);
  updateText(t);
}

// ------------------------------------------------------------------ pointer: seek and inspect

function timeAt(cv, clientX) {
  const r = cv.getBoundingClientRect();
  const [t0, , W] = laneWindow(engine.heard());
  return t0 + ((clientX - r.left) / r.width) * W;
}

function showTip(html, x, y) {
  const tip = $("#tip");
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(window.innerWidth - r.width - 8, x + 12) + "px";
  tip.style.top = Math.max(8, y - r.height - 10) + "px";
  clearTimeout(showTip.timer);
  showTip.timer = setTimeout(() => (tip.hidden = true), 4000);
}

function describeTime(s) {
  const bb = barBeat(s);
  return `${fmt(s, true)}${bb ? ` · bar ${bb.bar}, beat ${bb.beat}` : ""}`;
}

function attachPointer(tr) {
  const cv = tr.canvas;
  cv.addEventListener("click", (e) => {
    const r = cv.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    if (tr.hitBoxes) {
      for (const [x0, y0, x1, y1, n] of tr.hitBoxes) {
        if (px >= x0 - 2 && px <= x1 + 2 && py >= y0 - 2 && py <= y1 + 2) {
          showTip(`<b>${noteName(n[2])}</b> · ${esc(tr.label)}<br>from ${describeTime(n[0])}<br>to ${describeTime(n[1])} (${(n[1] - n[0]).toFixed(2)} s)`, e.clientX, e.clientY);
          return;
        }
      }
    }
    const t = timeAt(cv, e.clientX);
    if (tr.show === "chords" || (tr.show === "notes+chords" && py < 18)) {
      const i = segAt(tr.chords, t);
      if (i >= 0) {
        const [a, b, name] = tr.chords[i];
        showTip(`<b>${esc(name)}</b> · ${esc(tr.label)}<br>from ${describeTime(a)}<br>to ${describeTime(b)}`, e.clientX, e.clientY);
        return;
      }
    }
    engine.seek(t);
  });
}

function setupOverview() {
  const cv = $("#overview");
  let down = false;
  const go = (e) => {
    if (!state.data) return;
    const r = cv.getBoundingClientRect();
    engine.seek(((e.clientX - r.left) / r.width) * state.data.duration);
  };
  cv.addEventListener("pointerdown", (e) => { down = true; cv.setPointerCapture(e.pointerId); go(e); });
  cv.addEventListener("pointermove", (e) => { if (down) go(e); });
  cv.addEventListener("pointerup", () => (down = false));
}

function setupTransport() {
  $("#play").onclick = () => (engine.playing ? engine.pause() : engine.play());
  $("#back").onclick = () => engine.seek(engine.position() - 5);
  $("#fwd").onclick = () => engine.seek(engine.position() + 5);
  $("#mode-original").onclick = () => setMode("original");
  $("#mode-stems").onclick = () => setMode("stems");
  $("#zoom").oninput = (e) => (state.zoom = +e.target.value);
  $("#master").oninput = (e) => { state.master = +e.target.value; engine.setMaster(state.master); };
  document.addEventListener("keydown", (e) => {
    if ($("#player").hidden || e.target.closest("input,textarea") && e.target.type !== "range") return;
    if (e.code === "Space") { e.preventDefault(); $("#play").click(); }
    else if (e.code === "ArrowLeft") { e.preventDefault(); $("#back").click(); }
    else if (e.code === "ArrowRight") { e.preventDefault(); $("#fwd").click(); }
  });
}

// ------------------------------------------------------------------ start

async function start() {
  state.token = tokenFromUrl();
  setupUpload();
  setupTransport();
  setupOverview();
  try {
    await loadInfo();
  } catch (e) {
    $("#audio-status").hidden = false;
    $("#audio-status").textContent = "Cannot reach the server";
  }
  await refreshJobs();
  const m = location.hash.match(/song=([\w-]+)/);
  if (m && state.jobs.some((j) => j.id === m[1])) openJob(m[1]);
  requestAnimationFrame(frame);
}

start();
