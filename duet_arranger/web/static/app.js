/* Duet Arranger web app */
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

let info = null;
let lastState = null;
const plural = (n, one, many = one + "s") => `<b>${n}</b> ${n === 1 ? one : many}`;
let current = null; // job id on screen
let poller = null;
let token = "";
try {
  token = localStorage.getItem("duet.token") || "";
} catch {}

function authed(url) {
  if (!token) return url;
  return url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(token);
}

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (token) headers["X-Access-Token"] = token;
  const res = await fetch(path, { ...opts, headers });
  if (res.status === 401) {
    const t = prompt("This Duet Arranger asks for an access token:");
    if (t) {
      token = t.trim();
      try {
        localStorage.setItem("duet.token", token);
      } catch {}
      return api(path, opts);
    }
  }
  let data = null;
  try {
    data = await res.json();
  } catch {}
  if (!res.ok) throw Error(data?.detail || `Request failed (${res.status})`);
  return data;
}

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 4200);
}

/* --------------------------------------------------------------- the form */

function fillTuningSelects() {
  for (const id of ["tuning1", "tuning2"]) {
    const sel = $("#" + id);
    sel.innerHTML =
      `<option value="auto">Choose for me</option>` +
      info.tunings.map((t) => `<option value="${esc(t.name)}">${esc(t.name)} · ${esc(t.pitches)}</option>`).join("") +
      `<option value="custom">Custom…</option>`;
    sel.onchange = () => {
      $("#custom-row").hidden = !["tuning1", "tuning2"].some((k) => $("#" + k).value === "custom");
    };
  }
}

function formOptions() {
  const fd = new FormData();
  const tuning = (k, custom) => {
    const v = $("#" + k).value;
    return v === "custom" ? $("#" + custom).value.trim() || "auto" : v;
  };
  fd.set("tuning1", tuning("tuning1", "custom1"));
  fd.set("tuning2", tuning("tuning2", "custom2"));
  fd.set("capo", $("#capo").value);
  fd.set("percussion", $("#percussion").value);
  fd.set("max_fret", $("#max_fret").value || "20");
  fd.set("difficulty", $("#difficulty").value);
  fd.set("harmonics", $("#harmonics").checked ? "true" : "false");
  fd.set("taps", $("#taps").checked ? "true" : "false");
  fd.set("octaves", $("#octaves").checked ? "true" : "false");
  return fd;
}

function chooseFile(f) {
  if (!f) return;
  const dt = new DataTransfer();
  dt.items.add(f);
  $("#file").files = dt.files;
  $("#drop-title").textContent = f.name;
  $("#drop-sub").textContent = `${(f.size / 1048576).toFixed(1)} MB · ${/\.midi?$/i.test(f.name) ? "MIDI file" : "recording"}`;
  $("#drop").classList.add("chosen");
  if (!$("#title").value) $("#title").placeholder = f.name.replace(/\.[^.]+$/, "");
  $("#submit").disabled = false;
}

function bindForm() {
  const drop = $("#drop"); // a <label>: clicking it opens the file picker by itself
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      $("#file").click();
    }
  });
  $("#file").onchange = () => chooseFile($("#file").files[0]);
  ["dragenter", "dragover"].forEach((t) =>
    drop.addEventListener(t, (e) => {
      e.preventDefault();
      drop.classList.add("over");
    }),
  );
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    chooseFile(e.dataTransfer.files[0]);
  });
  $("#difficulty").oninput = () => ($("#difficulty-label").textContent = +$("#difficulty").value >= 0.95 ? "full" : +$("#difficulty").value >= 0.8 ? "medium" : "lighter");
  $("#new-form").onsubmit = async (e) => {
    e.preventDefault();
    const f = $("#file").files[0];
    if (!f) return;
    const fd = formOptions();
    fd.set("file", f);
    fd.set("title", $("#title").value.trim());
    fd.set("artist", $("#artist").value.trim());
    await start(() => api("/api/jobs", { method: "POST", body: fd }), "Uploading…");
  };
  $$("[data-demo]").forEach(
    (b) =>
      (b.onclick = async () => {
        const fd = formOptions();
        fd.set("style", b.dataset.demo);
        await start(() => api("/api/demo", { method: "POST", body: fd }));
      }),
  );
  $("#again").onsubmit = async (e) => {
    e.preventDefault();
    if (!current) return;
    const fd = formOptions();
    if ($("#title").value.trim()) fd.set("title", $("#title").value.trim());
    if ($("#artist").value.trim()) fd.set("artist", $("#artist").value.trim());
    await start(() => api(`/api/jobs/${current}/rearrange`, { method: "POST", body: fd }));
  };
}

async function start(request, busy = "Starting…") {
  const err = $("#form-error");
  err.hidden = true;
  const btn = $("#submit");
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = busy;
  try {
    const job = await request();
    await refreshHistory();
    show(job.id);
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  } finally {
    btn.textContent = label;
    btn.disabled = !$("#file").files[0];
  }
}

/* --------------------------------------------------------------- history */

const STATE_LABEL = { queued: "Waiting", running: "Arranging", done: "Ready", error: "Failed", cancelled: "Cancelled" };

function when(t) {
  const d = new Date(t * 1000);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + " · " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

async function refreshHistory() {
  let jobs = [];
  try {
    jobs = await api("/api/jobs");
  } catch {
    return;
  }
  $("#history-empty").hidden = jobs.length > 0;
  $("#history").innerHTML = jobs
    .map(
      (j) =>
        `<li><button class="history-item ${j.id === current ? "active" : ""}" data-job="${j.id}"><span class="dot ${j.state}"></span><span><strong>${esc(j.title || "Untitled")}</strong><small>${j.kind === "rearrange" ? "Re-arranged · " : ""}${STATE_LABEL[j.state] || j.state} · ${when(j.created)}</small></span></button></li>`,
    )
    .join("");
  $$("[data-job]").forEach((b) => (b.onclick = () => show(b.dataset.job)));
}

/* --------------------------------------------------------------- job view */

function show(id) {
  current = id;
  lastState = null;
  history.replaceState(null, "", "#" + id);
  $("#welcome").hidden = true;
  $("#job").hidden = false;
  $("#result").hidden = true;
  $("#atelier-frame-wrap").querySelector("iframe")?.remove();
  $("#atelier-hint").hidden = false;
  clearInterval(poller);
  poll();
  poller = setInterval(poll, 1000);
  $$("#history .history-item").forEach((b) => b.classList.toggle("active", b.dataset.job === id));
  if (window.innerWidth < 960) $("#job").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function poll() {
  const id = current;
  let job;
  try {
    job = await api(`/api/jobs/${id}`);
  } catch (e) {
    clearInterval(poller);
    toast(e.message);
    return;
  }
  if (id !== current) return;
  renderJob(job);
  if (job.state !== lastState) {
    lastState = job.state;
    refreshHistory();
  }
  if (!["queued", "running"].includes(job.state)) {
    clearInterval(poller);
    refreshHistory();
  }
}

function renderJob(job) {
  $("#job-title").textContent = job.title || "Untitled";
  $("#job-kind").textContent =
    { audio: "FROM A RECORDING", midi: "FROM A MIDI FILE", demo: "DEMO SONG", rearrange: "RE-ARRANGED" }[job.kind] || "ARRANGEMENT";
  $("#job-sub").textContent = [job.artist, job.input_name && job.kind !== "demo" ? job.input_name : ""].filter(Boolean).join(" · ");
  const st = $("#job-state");
  st.textContent = STATE_LABEL[job.state] || job.state;
  st.className = "state " + job.state;
  const running = ["queued", "running"].includes(job.state);
  $("#cancel").hidden = !running;
  $("#delete").hidden = running;
  $("#cancel").onclick = async () => {
    await api(`/api/jobs/${job.id}/cancel`, { method: "POST" });
    poll();
  };
  $("#delete").onclick = async () => {
    if (!confirm("Delete this arrangement and its files?")) return;
    await api(`/api/jobs/${job.id}`, { method: "DELETE" });
    current = null;
    history.replaceState(null, "", "#");
    $("#job").hidden = true;
    $("#welcome").hidden = false;
    refreshHistory();
  };
  $("#progress").hidden = !running;
  $("#failure").hidden = !["error", "cancelled"].includes(job.state);
  $("#failure-text").textContent = job.state === "cancelled" ? "You cancelled this arrangement." : job.error;
  if (running) renderProgress(job);
  if (job.state === "done" && $("#result").hidden) renderResult(job);
}

function renderProgress(job) {
  $("#bar-fill").style.width = Math.round(job.progress * 100) + "%";
  $("#progress-message").textContent = job.state === "queued" ? "Waiting for the previous arrangement to finish…" : job.message;
  $("#elapsed").textContent = job.elapsed ? `${Math.floor(job.elapsed / 60)}:${String(Math.floor(job.elapsed % 60)).padStart(2, "0")}` : "";
  const idx = job.stages.findIndex((s) => s.id === job.stage);
  $("#stages").innerHTML = job.stages
    .map((s, i) => {
      const state = i < idx ? "done" : i === idx ? "active" : "todo";
      const pct = i === idx ? ` <small>${Math.round(job.stage_progress * 100)}%</small>` : "";
      return `<li class="${state}"><span class="tick" aria-hidden="true">${state === "done" ? "✓" : i + 1}</span>${esc(s.label)}${pct}</li>`;
    })
    .join("");
}

const fileUrl = (job, name, download = false) => authed(`/api/jobs/${job.id}/files/${name.split("/").map(encodeURIComponent).join("/")}${download ? "?download=1" : ""}`);

function renderResult(job) {
  const r = job.result;
  $("#result").hidden = false;
  const made = job.version || r.version || "0.1";
  const old = made !== info.version;
  $("#outdated").hidden = !old;
  if (old)
    $("#outdated").innerHTML = `<strong>Made with an older Duet Arranger (${esc(made)}).</strong> This app is v${esc(info.version)}. Choose “Arrange again” below to rebuild it with the current version — the separated stems are reused.`;
  $("#facts").innerHTML = [
    ["Key", r.key],
    ["Tempo", r.tempo ? `${r.tempo} BPM` : "—"],
    ["Meter", r.meter],
    ["Length", `${r.bars} bars`],
    ["Style", r.percussive ? "Drum-driven · percussion on" : "Melodic · no percussion"],
  ]
    .map(([k, v]) => `<div class="fact"><small>${k}</small><strong>${esc(v)}</strong></div>`)
    .join("");
  const issues = r.issues || [];
  $("#audit").className = "audit " + (issues.length ? "warn" : "ok");
  $("#audit").innerHTML = issues.length
    ? `<strong>${issues.length} passage${issues.length > 1 ? "s" : ""} to check by hand</strong><ul>${issues.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>`
    : "<strong>✓ Every hand position checked</strong><span>No stretch, finger count, barre, tap or harmonic in either part is beyond a real pair of hands.</span>";
  $("#guitars").innerHTML = r.guitars
    .map(
      (g, i) => `<article class="guitar">
      <div class="guitar-head"><span class="roman">${["I", "II"][i] || i + 1}</span><div><strong>${esc(g.name)}</strong><small>${esc(g.role)}</small></div></div>
      <div class="tuning"><span>${esc(g.tuning)}</span><b>${esc(g.pitches)}</b><em>${g.capo ? "Capo " + g.capo : "No capo"}</em></div>
      <ul class="stats">
        <li>${plural(g.notes, "note")}</li>
        <li>${plural(g.open_strings, "open string")}</li>
        <li>${plural(g.harmonics, "harmonic")}</li>
        ${g.taps ? `<li>${plural(g.taps, "right-hand tap")}</li>` : ""}
        ${g.percussive ? `<li>${plural(g.percussive, "percussive hit")}</li>` : ""}
        ${g.dropped ? `<li>${plural(g.dropped, "note")} left out</li>` : ""}
      </ul></article>`,
    )
    .join("");
  const files = job.files || [];
  const main = files.find((f) => f.kind === "atelier");
  const downloads = files.filter((f) => ["atelier", "tab", "midi", "report"].includes(f.kind));
  $("#downloads").innerHTML = downloads.map((f) => `<a class="chip" href="${fileUrl(job, f.name, true)}" download>${esc(f.label)} ↓</a>`).join("");
  setupAtelier(job, main);
  // text tab
  const tab = files.find((f) => f.kind === "tab");
  $("#text-tab").textContent = "Loading…";
  if (tab)
    fetch(fileUrl(job, tab.name))
      .then((x) => x.text())
      .then((t) => ($("#text-tab").textContent = t));
  // stems
  const stems = files.filter((f) => f.kind === "stem" || f.kind === "original");
  $("#stems").innerHTML = stems.length
    ? stems
        .map(
          (f) =>
            `<div class="stem"><span>${esc(f.label)}</span><audio controls preload="none" src="${fileUrl(job, f.name)}"></audio><a class="chip" href="${fileUrl(job, f.name, true)}" download>↓</a></div>`,
        )
        .join("")
    : `<p class="muted">${job.kind === "audio" || job.parent ? "No stems were saved." : "Stems exist for recordings. This arrangement came from " + (job.kind === "demo" ? "the demo song." : "a MIDI file.")}</p>`;
  // tunings tried
  $("#alternatives").innerHTML = r.guitars
    .map(
      (g) =>
        `<h3>${esc(g.name)}</h3><table><thead><tr><th>Tuning</th><th>Capo</th><th>Effort / note</th><th>Open-string resonance</th><th>Left out</th><th>Score</th></tr></thead><tbody>${g.considered
          .map(
            (c) =>
              `<tr class="${c.tuning === g.tuning && c.capo === g.capo ? "chosen" : ""}"><td>${esc(c.tuning)} <small>${esc(c.pitches)}</small></td><td>${c.capo || "—"}</td><td>${c.cost_per_note}</td><td>${c.open_string_resonance_notes ?? "—"}</td><td>${c.dropped}</td><td>${c.score}</td></tr>`,
          )
          .join("")}</tbody></table>`,
    )
    .join("") + `<p class="muted">Lower score wins. A capo has to save a large share of the effort before it is used.</p>`;
  $("#song-info").innerHTML =
    `<h3>Sections</h3><p>${r.sections.map((s) => `<span class="chip">${esc(s.label)} · bar ${s.bar}</span>`).join(" ") || "—"}</p>` +
    `<h3>Chords heard</h3><p>${r.chords.map((c) => `<span class="chip">${esc(c)}</span>`).join(" ") || "—"}</p>` +
    `<p class="muted">Drum prominence ${Math.round(r.drum_prominence * 100)}%. ${r.percussive ? "The drums drive this song, so Guitar II adds kick and snare percussion." : "The drums are not a main part of this song, so no percussion was written."}</p>`;
  selectTab("atelier");
}

function selectTab(name) {
  $$(".tabs [data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".panel").forEach((p) => (p.hidden = p.dataset.panel !== name));
}

/* ------------------------------------------------------- Atelier hand-off */

const pending = new Map(); // window -> {job, file}
window.addEventListener("message", async (ev) => {
  if (ev.data?.type !== "atelier:ready") return;
  const want = pending.get(ev.source);
  if (!want) return;
  pending.delete(ev.source);
  try {
    const project = await (await fetch(fileUrl(want.job, want.file.name))).json();
    ev.source.postMessage({ type: "atelier:import", project, name: want.job.title }, new URL(info.atelier_url, location.href).origin);
  } catch (e) {
    toast("Could not send the duet to Atelier: " + e.message);
  }
});

function atelierUrl(params) {
  const u = new URL(info.atelier_url, location.href);
  Object.entries(params).forEach(([k, v]) => u.searchParams.set(k, v));
  return u.toString();
}

function setupAtelier(job, file) {
  const open = $("#open-atelier");
  if (!file) {
    open.hidden = true;
    return;
  }
  open.hidden = false;
  if (info.atelier_local) {
    // same origin: Atelier fetches the project itself
    open.href = atelierUrl({ import: fileUrl(job, file.name) });
    open.onclick = null;
  } else {
    open.href = info.atelier_url;
    open.onclick = (e) => {
      e.preventDefault();
      const w = window.open(atelierUrl({ from: "duet-arranger" }), "atelier-duet");
      if (!w) return toast("Allow pop-ups for this page, or download the Atelier file and import it.");
      pending.set(w, { job, file });
      toast("Opening Atelier… the duet arrives as soon as it has loaded.");
    };
  }
  $("#show-here").onclick = () => {
    selectTab("atelier");
    const wrap = $("#atelier-frame-wrap");
    wrap.querySelector("iframe")?.remove();
    $("#atelier-hint").hidden = true;
    const frame = document.createElement("iframe");
    frame.title = "Atelier · " + job.title;
    frame.allow = "autoplay";
    const params = { embed: "1" };
    if (info.atelier_local) params.import = fileUrl(job, file.name);
    frame.src = atelierUrl(params);
    wrap.appendChild(frame);
    if (!info.atelier_local) frame.addEventListener("load", () => pending.set(frame.contentWindow, { job, file }), { once: true });
    if (!info.atelier_local) pending.set(frame.contentWindow, { job, file });
    frame.scrollIntoView({ behavior: "smooth", block: "start" });
  };
}

/* ------------------------------------------------------------------ boot */

async function boot() {
  try {
    info = await api("/api/info");
  } catch (e) {
    toast(e.message);
    return;
  }
  $("#version").textContent = "v" + info.version;
  fillTuningSelects();
  bindForm();
  $$(".tabs [data-tab]").forEach((b) => (b.onclick = () => selectTab(b.dataset.tab)));
  $("#atelier-link").href = info.atelier_url;
  $("#file").accept = info.accept.join(",");
  const pill = $("#audio-status");
  if (!info.audio.available) {
    pill.hidden = false;
    pill.className = "pill warn";
    pill.textContent = "MIDI only · audio needs " + info.audio.missing.join(", ");
    pill.title = "pip install -r requirements-audio.txt && pip install --no-deps basic-pitch";
  }
  await refreshHistory();
  const fromHash = () => {
    const id = location.hash.slice(1);
    if (/^[a-f0-9]{16}$/.test(id) && id !== current) show(id);
  };
  window.addEventListener("hashchange", fromHash);
  fromHash();
}
boot();
