(() => {
  "use strict";
  const MAX_BYTES = 4 * 1024 * 1024;
  const ACCEPTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
  const $ = (id) => document.getElementById(id);
  const el = {
    serviceStatus: $("serviceStatus"), serviceStatusText: $("serviceStatusText"),
    modelSubline: $("modelSubline"), dropzone: $("dropzone"), fileInput: $("fileInput"),
    selectedFile: $("selectedFile"), fileName: $("fileName"), fileSize: $("fileSize"),
    btnReset: $("btnReset"), btnDemo: $("btnDemo"), btnDetect: $("btnDetect"),
    progressBlock: $("progressBlock"), progressTitle: $("progressTitle"),
    progressMeta: $("progressMeta"), progressTrack: $("progressTrack"),
    progressFill: $("progressFill"), progressDescription: $("progressDescription"),
    errorNotice: $("errorNotice"), errorText: $("errorText"),
    resultState: $("resultState"), emptyResult: $("emptyResult"), canvasWrap: $("canvasWrap"),
    canvas: $("resultCanvas"), resultMetrics: $("resultMetrics"), metricCount: $("metricCount"),
    metricConfidence: $("metricConfidence"), metricLatency: $("metricLatency"),
    detectionList: $("detectionList")
  };
  let selectedFile = null, objectUrl = null, modelReady = false, analysing = false, lastResult = null;

  function setStatus(status, message) {
    el.serviceStatus.dataset.status = status;
    el.serviceStatusText.textContent = message;
    el.modelSubline.textContent = status === "ready" ? "Модель готова" : status === "error" ? "Нужна проверка" : "Подготовка модели…";
  }
  function refreshButtons() {
    el.btnDetect.disabled = !selectedFile || !modelReady || analysing;
    el.btnDemo.disabled = analysing;
    el.btnReset.disabled = analysing;
    el.fileInput.disabled = analysing;
  }
  function showError(message) {
    el.errorText.textContent = message;
    el.errorNotice.hidden = false;
  }
  function clearError() { el.errorNotice.hidden = true; el.errorText.textContent = ""; }
  function setResultState(text, state) {
    el.resultState.textContent = text;
    el.resultState.dataset.state = state || "default";
  }
  function progress(title, meta, description, percent) {
    el.progressBlock.hidden = false;
    el.progressTitle.textContent = title;
    el.progressMeta.textContent = meta;
    el.progressDescription.textContent = description;
    if (percent === null) {
      el.progressTrack.classList.add("indeterminate");
      el.progressFill.style.width = "38%";
    } else {
      el.progressTrack.classList.remove("indeterminate");
      el.progressFill.style.width = Math.max(0, Math.min(100, percent)) + "%";
    }
  }
  function humanBytes(bytes) {
    return bytes < 1024 * 1024 ? Math.max(1, Math.round(bytes / 1024)) + " KB" : (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }
  function acceptFile(file) {
    if (!file) return;
    clearError();
    if (!ACCEPTED_TYPES.has(file.type)) return showError("Подходит только изображение JPEG, PNG или WebP.");
    if (file.size > MAX_BYTES) return showError("Файл больше 4 MB. Выбери изображение поменьше.");
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    selectedFile = file;
    objectUrl = URL.createObjectURL(file);
    el.fileName.textContent = file.name || "image-from-clipboard";
    el.fileSize.textContent = humanBytes(file.size);
    el.selectedFile.hidden = false;
    el.dropzone.hidden = true;
    el.progressBlock.hidden = true;
    lastResult = null;
    el.resultMetrics.hidden = true;
    el.detectionList.hidden = true;
    setResultState("Готово к анализу");
    drawPreview(objectUrl);
    refreshButtons();
  }
  function drawPreview(src) {
    const image = new Image();
    image.onload = () => {
      el.canvas.width = image.naturalWidth;
      el.canvas.height = image.naturalHeight;
      const ctx = el.canvas.getContext("2d");
      ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
      ctx.drawImage(image, 0, 0);
      el.emptyResult.hidden = true;
      el.canvasWrap.hidden = false;
    };
    image.onerror = () => showError("Не удалось показать выбранное изображение.");
    image.src = src;
  }
  function drawDetections(result) {
    const image = new Image();
    image.onload = () => {
      const canvas = el.canvas, ctx = canvas.getContext("2d");
      canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
      ctx.drawImage(image, 0, 0);
      const factor = Math.max(canvas.width, canvas.height) / 900;
      const lineWidth = Math.max(2, factor * 2.1);
      const fontSize = Math.max(12, Math.min(22, factor * 17));
      ctx.lineWidth = lineWidth; ctx.font = "650 " + fontSize + "px system-ui, sans-serif"; ctx.textBaseline = "middle";
      (result.detections || []).forEach((d) => {
        const box = Array.isArray(d.bbox_xyxy) ? d.bbox_xyxy.map(Number) : [];
        if (box.length !== 4 || !box.every(Number.isFinite)) return;
        const x = box[0], y = box[1], w = Math.max(0, box[2] - box[0]), h = Math.max(0, box[3] - box[1]);
        ctx.strokeStyle = "#f2a64b"; ctx.fillStyle = "rgba(242,166,75,.08)";
        ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
        const label = "Молоток · " + Math.round((Number(d.conf) || 0) * 100) + "%";
        const labelW = ctx.measureText(label).width + 14, labelH = fontSize + 10;
        const labelY = Math.max(0, y - labelH - 2);
        ctx.fillStyle = "#f2a64b"; ctx.fillRect(Math.max(0, x), labelY, labelW, labelH);
        ctx.fillStyle = "#242a32"; ctx.fillText(label, Math.max(0, x) + 7, labelY + labelH / 2);
      });
      el.emptyResult.hidden = true; el.canvasWrap.hidden = false;
    };
    image.onerror = () => showError("Не удалось отобразить результат.");
    image.src = objectUrl;
  }
  function renderResults(result) {
    lastResult = result;
    const detections = Array.isArray(result.detections) ? result.detections : [];
    const maxConfidence = detections.reduce((max, item) => Math.max(max, Number(item.conf) || 0), 0);
    el.metricCount.textContent = String(detections.length);
    el.metricConfidence.textContent = detections.length ? Math.round(maxConfidence * 100) + "%" : "—";
    el.metricLatency.textContent = Math.round(Number(result.inference_ms) || Number(result.inference_sec) * 1000 || 0) + " мс";
    el.resultMetrics.hidden = false;
    el.detectionList.replaceChildren();
    detections.forEach((item, index) => {
      const row = document.createElement("div"); row.className = "detection-item";
      const swatch = document.createElement("span"); swatch.className = "detection-swatch";
      const label = document.createElement("span"); label.className = "detection-label";
      label.textContent = (item.class === "hammer" ? "Молоток" : item.class || "Объект") + " " + (index + 1);
      const confidence = document.createElement("span"); confidence.className = "detection-confidence";
      confidence.textContent = Math.round((Number(item.conf) || 0) * 100) + "% уверенности";
      row.append(swatch, label, confidence); el.detectionList.append(row);
    });
    el.detectionList.hidden = detections.length === 0;
    setResultState(detections.length ? "Найдено" : "Без обнаружений", detections.length ? "success" : "empty");
    drawDetections(result);
  }
  function friendlyError(status, data) {
    if (status === 413) return "Файл больше допустимого лимита 4 MB.";
    if (status === 415) return "Формат не поддерживается. Используй JPEG, PNG или WebP.";
    if (status === 422) return "Сервер не получил файл. Выбери изображение и повтори запрос.";
    if (status === 503) return "Модель ещё не готова. Дождись статуса «Модель готова» и повтори запрос.";
    if (status >= 500) return "Ошибка сервера" + (data && data.request_id ? " · request_id: " + data.request_id : "") + ". Попробуй ещё раз.";
    return (data && (data.error || data.message)) || "Не удалось обработать изображение.";
  }
  function finishAnalysis() {
    analysing = false; refreshButtons();
    if (lastResult) {
      el.progressMeta.textContent = "Готово";
      el.progressTrack.classList.remove("indeterminate");
    } else {
      el.progressMeta.textContent = "Не завершено";
      el.progressTrack.classList.remove("indeterminate");
    }
  }
  function analyse() {
    if (!selectedFile || !modelReady || analysing) return;
    clearError(); lastResult = null; el.resultMetrics.hidden = true; el.detectionList.hidden = true;
    drawPreview(objectUrl); analysing = true; refreshButtons(); setResultState("Обработка…");
    progress("Отправка изображения", "0%", "Передаём выбранный файл в API.", 0);
    const form = new FormData(); form.append("file", selectedFile, selectedFile.name || "image.png");
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/detect", true); xhr.timeout = 180000; xhr.setRequestHeader("Accept", "application/json");
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return progress("Отправка изображения", "Передача", "Загружаем файл.", null);
      const percent = Math.round(event.loaded / event.total * 100);
      progress("Отправка изображения", percent + "%", "Реальный прогресс передачи файла.", percent);
    };
    xhr.upload.onload = () => progress("Модель анализирует изображение", "Инференс", "Файл передан. YOLO26n выполняет вычисления на CPU.", null);
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText || "{}"); } catch (_) { /* invalid response handled below */ }
      if (xhr.status >= 200 && xhr.status < 300 && Array.isArray(data.detections)) {
        progress("Анализ завершён", "Готово", "Запрос " + (data.request_id || "выполнен") + ". Получен ответ модели.", 100);
        renderResults(data);
      } else {
        showError(friendlyError(xhr.status, data)); setResultState("Ошибка анализа");
        if (data.request_id) el.progressDescription.textContent = "Идентификатор запроса: " + data.request_id;
      }
      finishAnalysis();
    };
    xhr.onerror = () => { showError("Нет соединения с API. На Render после простоя сервис может запускаться около минуты."); setResultState("Нет соединения"); finishAnalysis(); };
    xhr.ontimeout = () => { showError("Анализ превысил время ожидания. Проверь состояние сервиса и попробуй ещё раз."); finishAnalysis(); };
    xhr.onabort = finishAnalysis;
    xhr.send(form);
  }
  async function pollReadiness() {
    let failures = 0, attempt = 0;
    const maxAttempts = 100;
    while (attempt < maxAttempts) {
      attempt++;
      try {
        const response = await fetch("/ready", { cache: "no-store", headers: { "Accept": "application/json" } });
        const data = await response.json();
        if (response.ok && data.status === "ready" && data.warmed_up) {
          modelReady = true; setStatus("ready", "Модель готова"); clearError(); refreshButtons();
          if (!selectedFile) setResultState("Готова к анализу");
          el.progressBlock.hidden = true;
          return;
        }
        failures = 0;
        modelReady = false;
        if (data.stage === "loading_model") {
          setStatus("loading", "Загрузка модели");
          progress("Загрузка модели", "Этап 1 из 2", "Создаём сессию ONNX Runtime на CPU.", null);
        } else if (data.stage === "warming_up") {
          setStatus("loading", "Прогрев модели");
          progress("Прогрев модели", "Этап 2 из 2", "Выполняется пробный инференс перед приёмом запросов.", null);
        } else if (data.stage === "error") {
          setStatus("error", "Модель не готова");
          progress("Модель не запущена", "Нужна проверка", "Проверь наличие best.onnx и логи сервера.", null);
        } else {
          setStatus("loading", "Подключение к сервису");
          progress("Подключение к сервису", "Ожидание", "Проверяем доступность модели.", null);
        }
        refreshButtons();
      } catch (_) {
        failures++; modelReady = false; setStatus("loading", "Ожидание сервиса");
        progress("Ожидание сервиса", "Подключение", "Render может запускать контейнер после простоя.", null);
        refreshButtons();
        if (failures >= 5) showError("Сервис пока не отвечает. Проверь URL или повтори попытку через несколько секунд.");
      }
      await new Promise((resolve) => setTimeout(resolve, 2500));
    }
    setStatus("error", "Время ожидания истекло");
    showError("Не удалось дождаться готовности модели. Обнови страницу и проверь логи сервиса.");
  }
  function resetSelection() {
    if (analysing) return;
    selectedFile = null; lastResult = null;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null; el.fileInput.value = ""; el.selectedFile.hidden = true; el.dropzone.hidden = false;
    el.canvasWrap.hidden = true; el.emptyResult.hidden = false; el.resultMetrics.hidden = true; el.detectionList.hidden = true;
    el.progressBlock.hidden = true; clearError(); setResultState("Ожидает фото"); refreshButtons();
  }

  el.dropzone.addEventListener("click", () => el.fileInput.click());
  el.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); el.fileInput.click(); }
  });
  el.fileInput.addEventListener("change", (event) => acceptFile(event.target.files && event.target.files[0]));
  el.dropzone.addEventListener("dragover", (event) => { event.preventDefault(); el.dropzone.classList.add("dragover"); });
  el.dropzone.addEventListener("dragleave", () => el.dropzone.classList.remove("dragover"));
  el.dropzone.addEventListener("drop", (event) => {
    event.preventDefault(); el.dropzone.classList.remove("dragover");
    acceptFile(event.dataTransfer && event.dataTransfer.files[0]);
  });
  document.addEventListener("paste", (event) => {
    const items = Array.from((event.clipboardData && event.clipboardData.items) || []);
    const item = items.find((entry) => entry.kind === "file" && ACCEPTED_TYPES.has(entry.type));
    if (item) {
      const file = item.getAsFile();
      if (file) acceptFile(new File([file], "clipboard." + (file.type.split("/")[1] || "png"), { type: file.type }));
    }
  });
  el.btnReset.addEventListener("click", resetSelection);
  el.btnDetect.addEventListener("click", analyse);
  el.btnDemo.addEventListener("click", async () => {
    clearError(); el.btnDemo.disabled = true;
    try {
      const response = await fetch("/static/img/demo.jpg", { cache: "no-store" });
      if (!response.ok) throw new Error("demo_missing");
      const blob = await response.blob();
      acceptFile(new File([blob], "demo.jpg", { type: blob.type || "image/jpeg" }));
    } catch (_) {
      showError("Не найден static/img/demo.jpg. Сохрани демонстрационное фото из исходного проекта.");
    } finally { el.btnDemo.disabled = false; }
  });
  window.addEventListener("beforeunload", () => { if (objectUrl) URL.revokeObjectURL(objectUrl); });
  refreshButtons();
  pollReadiness();
})();