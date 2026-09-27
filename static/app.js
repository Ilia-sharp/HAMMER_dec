"use strict";

const $ = (id) => document.getElementById(id);
const MAX_MB = 4;
const TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

const els = {
  dropzone: $("dropzone"), fileInput: $("fileInput"),
  btnDetect: $("btnDetect"), btnDemo: $("btnDemo"), btnReset: $("btnReset"),
  errBanner: $("errBanner"), progressCard: $("progressCard"), pbarFill: $("pbarFill"),
  resultSection: $("resultSection"), canvasWrap: $("canvasWrap"),
  preview: $("preview"), overlay: $("overlay"), statLine: $("statLine"),
};

let currentFile = null;
let busy = false;
let objectUrl = null;

function showError(msg) { els.errBanner.textContent = msg; els.errBanner.hidden = false; }
function hideError() { els.errBanner.hidden = true; }

function setFile(file) {
  if (!file) return;
  if (!TYPES.has(file.type)) return showError("Поддерживаются только JPEG, PNG и WebP.");
  if (file.size > MAX_MB * 1024 * 1024) return showError(`Файл больше ${MAX_MB} МБ.`);

  hideError();
  clearOverlay();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);

  els.preview.onload = () => { els.resultSection.hidden = false; sizeOverlay(); requestAnimationFrame(fitCanvasCss); };
  els.preview.src = objectUrl;

  currentFile = file;
  els.btnDetect.disabled = false;
  els.btnReset.hidden = false;
  els.statLine.textContent = "";
}

function resetAll() {
  currentFile = null;
  clearOverlay();
  if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
  els.preview.removeAttribute("src");
  els.resultSection.hidden = true;
  els.btnDetect.disabled = true;
  els.btnReset.hidden = true;
  hideError();
}

els.dropzone.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); }
});
els.fileInput.addEventListener("change", () => setFile(els.fileInput.files[0]));
els.btnReset.addEventListener("click", resetAll);

["dragenter", "dragover"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("drag"); }));
els.dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

window.addEventListener("paste", (e) => {
  const f = [...(e.clipboardData?.files || [])].find((x) => TYPES.has(x.type));
  if (f) setFile(f);
});

els.btnDemo.addEventListener("click", async () => {
  try {
    const r = await fetch("/static/img/demo.jpg");
    if (!r.ok) throw new Error();
    setFile(new File([await r.blob()], "demo.jpg", { type: "image/jpeg" }));
  } catch {
    showError("Не удалось загрузить пример.");
  }
});

/* ── Детекция ──────────────────────────────────────── */
els.btnDetect.addEventListener("click", detect);

async function detect() {
  if (!currentFile || busy) return;
  busy = true;
  hideError();
  els.btnDetect.disabled = true;
  els.canvasWrap.classList.add("scanning");
  els.progressCard.hidden = false;
  els.pbarFill.style.width = "10%";

  const timer = setInterval(() => {
    const w = parseFloat(els.pbarFill.style.width) || 0;
    els.pbarFill.style.width = Math.min(90, w + 6) + "%";
  }, 150);

  const fd = new FormData();
  fd.append("file", currentFile, currentFile.name);

  try {
    const r = await fetch("/detect", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);

    await new Promise((res) => {
      if (els.preview.complete && els.preview.naturalWidth) res();
      else els.preview.onload = res;
    });
    els.pbarFill.style.width = "100%";
    drawDetections(j);
    renderStats(j);
  } catch (e) {
    showError("Ошибка детекции: " + e.message);
  } finally {
    clearInterval(timer);
    setTimeout(() => { els.progressCard.hidden = true; els.pbarFill.style.width = "0%"; }, 300);
    els.canvasWrap.classList.remove("scanning");
    els.btnDetect.disabled = !currentFile;
    busy = false;
  }
}

/* ── Отрисовка рамок ───────────────────────────────── */
function sizeOverlay() {
  const { naturalWidth: w, naturalHeight: h } = els.preview;
  if (!w) return;
  els.overlay.width = w;
  els.overlay.height = h;
}

function fitCanvasCss() {
  const img = els.preview;
  if (!img.clientWidth) return;
  els.overlay.style.width = img.clientWidth + "px";
  els.overlay.style.height = img.clientHeight + "px";
}
window.addEventListener("resize", () => { if (!els.resultSection.hidden) fitCanvasCss(); });

function clearOverlay() {
  const ctx = els.overlay.getContext("2d");
  ctx && ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
}

function drawDetections(j) {
  sizeOverlay();
  fitCanvasCss();
  const ctx = els.overlay.getContext("2d");
  const W = els.overlay.width, H = els.overlay.height;
  ctx.clearRect(0, 0, W, H);

  const inf = j.image || {};
  const k = inf.inference_width ? W / inf.inference_width : 1;
  const color = "#f59e0b";

  j.detections.forEach((d) => {
    const [x1, y1, x2, y2] = d.bbox_xyxy.map((v) => v * k);
    const lw = Math.max(2, Math.round(W / 300));
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${d.class} ${d.conf.toFixed(2)}`;
    const fs = Math.max(13, Math.round(W / 42));
    ctx.font = `700 ${fs}px system-ui, sans-serif`;
    const tw = ctx.measureText(label).width;
    const above = y1 > 30;
    const ly = above ? y1 - 24 : y1;
    ctx.fillStyle = color;
    ctx.fillRect(x1, ly, tw + 10, 24);
    ctx.fillStyle = "#17110a";
    ctx.fillText(label, x1 + 5, ly + 17);
  });
}

function renderStats(j) {
  if (!j.count) {
    els.statLine.innerHTML = "Молоток не найден — попробуйте другой ракурс.";
    return;
  }
  const best = Math.max(...j.detections.map((d) => d.conf));
  els.statLine.innerHTML =
    `найдено: <b>${j.count}</b> · уверенность: <b>${best.toFixed(2)}</b> · время: <b>${j.inference_sec.toFixed(2)} c</b>`;
}
