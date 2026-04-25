const RAW_W = 1280;
const RAW_H = 720;
const VIDEO_W = 544;
const VIDEO_H = 306;
const SCALED_W = 306;
const SCALED_H = 306;

const imageInput = document.getElementById("imageInput");
const videoInput1280 = document.getElementById("videoInput1280");
const videoInput306 = document.getElementById("videoInput306");
const dropZone = document.getElementById("dropZone");
const imageList = document.getElementById("imageList");
const frameImage = document.getElementById("frameImage");
const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");
const emptyCanvas = document.getElementById("emptyCanvas");
const currentBoxes = document.getElementById("currentBoxes");
const clearFrame = document.getElementById("clearFrame");
const coordsRaw = document.getElementById("coordsRaw");
const coords544 = document.getElementById("coords544");
const coords306 = document.getElementById("coords306");
const statusText = document.getElementById("statusText");
const testVideo = document.getElementById("testVideo");
const videoOverlay = document.getElementById("videoOverlay");
const emptyVideo = document.getElementById("emptyVideo");
const videoStage = document.querySelector(".video-stage");

let state = { images: [], boxes_by_image: {} };
let activeImageId = null;
let selectedBoxId = null;
let draftBox = null;
let isDragging = false;
let dragStart = null;
let videoMaskMode = { width: RAW_W, height: RAW_H, label: "1280x720" };

function activeBoxes() {
  return activeImageId ? state.boxes_by_image[activeImageId] || [] : [];
}

function rawPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(RAW_W, ((event.clientX - rect.left) / rect.width) * RAW_W)),
    y: Math.max(0, Math.min(RAW_H, ((event.clientY - rect.top) / rect.height) * RAW_H)),
  };
}

function normalizeBox(box) {
  const x = Math.min(box.x, box.x + box.w);
  const y = Math.min(box.y, box.y + box.h);
  return {
    id: box.id || `box_${Date.now()}_${Math.random().toString(16).slice(2)}`,
    x: Math.round(Math.max(0, Math.min(RAW_W, x))),
    y: Math.round(Math.max(0, Math.min(RAW_H, y))),
    w: Math.round(Math.abs(box.w)),
    h: Math.round(Math.abs(box.h)),
  };
}

function scaledBox(box, width, height) {
  return {
    x: Number(((box.x * width) / RAW_W).toFixed(3)),
    y: Number(((box.y * height) / RAW_H).toFixed(3)),
    w: Number(((box.w * width) / RAW_W).toFixed(3)),
    h: Number(((box.h * height) / RAW_H).toFixed(3)),
  };
}

function unionBoxes() {
  return state.images.flatMap((image) =>
    (state.boxes_by_image[image.id] || []).map((box) => ({
      ...box,
      image_id: image.id,
      image_name: image.name,
    })),
  );
}

function drawBox(box, options = {}) {
  ctx.save();
  ctx.lineWidth = options.selected ? 4 : 2;
  ctx.strokeStyle = options.draft ? "#f59e0b" : options.selected ? "#22c55e" : "#14b8a6";
  ctx.fillStyle = options.draft ? "rgba(245, 158, 11, 0.18)" : "rgba(20, 184, 166, 0.15)";
  ctx.fillRect(box.x, box.y, box.w, box.h);
  ctx.strokeRect(box.x, box.y, box.w, box.h);
  ctx.fillStyle = "#111827";
  ctx.fillRect(box.x, Math.max(0, box.y - 20), 82, 18);
  ctx.fillStyle = "#ffffff";
  ctx.font = "13px Arial";
  ctx.fillText(`${box.x},${box.y}`, box.x + 5, Math.max(14, box.y - 6));
  ctx.restore();
}

function redraw() {
  ctx.clearRect(0, 0, RAW_W, RAW_H);
  activeBoxes().forEach((box) => drawBox(box, { selected: box.id === selectedBoxId }));
  if (draftBox) {
    drawBox(normalizeBox(draftBox), { draft: true });
  }
}

function renderImages() {
  imageList.innerHTML = "";
  state.images.forEach((image) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `image-item${image.id === activeImageId ? " active" : ""}`;
    button.textContent = `${image.name} (${(state.boxes_by_image[image.id] || []).length})`;
    button.addEventListener("click", () => setActiveImage(image.id));
    imageList.appendChild(button);
  });
}

function renderCurrentBoxes() {
  currentBoxes.innerHTML = "";
  activeBoxes().forEach((box, index) => {
    const row = document.createElement("div");
    row.className = `box-item${box.id === selectedBoxId ? " active" : ""}`;

    const label = document.createElement("button");
    label.type = "button";
    label.className = "box-select";
    label.textContent = `${index + 1}`;
    label.title = "Select box";
    label.addEventListener("click", () => {
      selectedBoxId = box.id;
      renderAll();
    });
    row.appendChild(label);

    ["x", "y", "w", "h"].forEach((field) => {
      const wrap = document.createElement("label");
      wrap.className = "box-field";
      wrap.textContent = field;

      const input = document.createElement("input");
      input.type = "number";
      input.inputMode = "numeric";
      input.step = "1";
      input.value = box[field];
      input.min = field === "w" || field === "h" ? "1" : "0";
      input.max = field === "x" || field === "w" ? RAW_W : RAW_H;
      input.addEventListener("focus", () => {
        selectedBoxId = box.id;
        redraw();
        renderImages();
        renderCoords();
        renderVideoOverlay();
        row.classList.add("active");
      });
      input.addEventListener("change", async () => {
        const nextValue = Number(input.value);
        if (!Number.isFinite(nextValue)) {
          input.value = box[field];
          return;
        }
        box[field] = Math.round(nextValue);
        const normalized = normalizeBox(box);
        Object.assign(box, normalized);
        await persistActiveBoxes();
        renderAll();
      });
      wrap.appendChild(input);
      row.appendChild(wrap);
    });

    currentBoxes.appendChild(row);
  });
}

function renderCoords() {
  const raw = unionBoxes().map(({ x, y, w, h }) => ({ x, y, w, h }));
  coordsRaw.textContent = JSON.stringify(raw, null, 2);
  coords544.textContent = JSON.stringify(raw.map((box) => scaledBox(box, VIDEO_W, VIDEO_H)), null, 2);
  coords306.textContent = JSON.stringify(raw.map((box) => scaledBox(box, SCALED_W, SCALED_H)), null, 2);
}

function renderVideoOverlay() {
  videoOverlay.innerHTML = "";
  unionBoxes().forEach((box) => {
    const maskBox = scaledBox(box, videoMaskMode.width, videoMaskMode.height);
    const div = document.createElement("div");
    div.className = "mask-block";
    div.style.left = `${(maskBox.x / videoMaskMode.width) * 100}%`;
    div.style.top = `${(maskBox.y / videoMaskMode.height) * 100}%`;
    div.style.width = `${(maskBox.w / videoMaskMode.width) * 100}%`;
    div.style.height = `${(maskBox.h / videoMaskMode.height) * 100}%`;
    videoOverlay.appendChild(div);
  });
}

function setVideoMaskMode(width, height, label) {
  videoMaskMode = { width, height, label };
  videoStage.style.aspectRatio = `${width} / ${height}`;
  statusText.textContent = `Video preview is using the ${label} mask coordinates.`;
  renderVideoOverlay();
}

function renderAll() {
  renderImages();
  renderCurrentBoxes();
  renderCoords();
  renderVideoOverlay();
  redraw();
}

function setActiveImage(imageId) {
  activeImageId = imageId;
  selectedBoxId = null;
  const image = state.images.find((item) => item.id === imageId);
  if (image) {
    frameImage.src = `/images/${image.filename}`;
    emptyCanvas.style.display = "none";
  }
  renderAll();
}

async function loadState() {
  const response = await fetch("/api/state");
  state = await response.json();
  if (!activeImageId && state.images.length > 0) {
    activeImageId = state.images[0].id;
    frameImage.src = `/images/${state.images[0].filename}`;
    emptyCanvas.style.display = "none";
  }
  renderAll();
}

async function persistActiveBoxes() {
  if (!activeImageId) return;
  await fetch(`/api/images/${activeImageId}/boxes`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ boxes: activeBoxes() }),
  });
}

async function uploadImageFile(file) {
  const form = new FormData();
  form.append("image", file);
  const response = await fetch("/api/images", { method: "POST", body: form });
  if (!response.ok) {
    statusText.textContent = await response.text();
    return;
  }
  const image = await response.json();
  await loadState();
  setActiveImage(image.id);
}

async function uploadPastedImage(file) {
  const dataUrl = await new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(file);
  });
  const response = await fetch("/api/images", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: `pasted_${new Date().toISOString().replace(/[:.]/g, "-")}.png`, data_url: dataUrl }),
  });
  const image = await response.json();
  await loadState();
  setActiveImage(image.id);
}

function hitTest(point) {
  return [...activeBoxes()].reverse().find(
    (box) => point.x >= box.x && point.x <= box.x + box.w && point.y >= box.y && point.y <= box.y + box.h,
  );
}

canvas.addEventListener("mousedown", (event) => {
  if (!activeImageId) return;
  const point = rawPoint(event);
  const hit = hitTest(point);
  if (hit) {
    selectedBoxId = hit.id;
    renderAll();
    return;
  }
  isDragging = true;
  dragStart = point;
  draftBox = { x: point.x, y: point.y, w: 0, h: 0 };
});

canvas.addEventListener("mousemove", (event) => {
  if (!isDragging || !draftBox) return;
  const point = rawPoint(event);
  draftBox = { x: dragStart.x, y: dragStart.y, w: point.x - dragStart.x, h: point.y - dragStart.y };
  redraw();
});

window.addEventListener("mouseup", async () => {
  if (!isDragging || !draftBox) return;
  const box = normalizeBox(draftBox);
  isDragging = false;
  draftBox = null;
  if (box.w >= 3 && box.h >= 3) {
    state.boxes_by_image[activeImageId].push(box);
    selectedBoxId = box.id;
    await persistActiveBoxes();
  }
  renderAll();
});

window.addEventListener("keydown", async (event) => {
  if (event.key !== "Delete" && event.key !== "Backspace") return;
  if (!activeImageId || !selectedBoxId) return;
  state.boxes_by_image[activeImageId] = activeBoxes().filter((box) => box.id !== selectedBoxId);
  selectedBoxId = null;
  await persistActiveBoxes();
  renderAll();
});

clearFrame.addEventListener("click", async () => {
  if (!activeImageId) return;
  state.boxes_by_image[activeImageId] = [];
  selectedBoxId = null;
  await persistActiveBoxes();
  renderAll();
});

imageInput.addEventListener("change", async () => {
  for (const file of imageInput.files) {
    await uploadImageFile(file);
  }
  imageInput.value = "";
});

async function uploadTestVideo(file, width, height, label) {
  if (!file) return;
  const form = new FormData();
  form.append("video", file);
  const response = await fetch("/api/videos", { method: "POST", body: form });
  const video = await response.json();
  setVideoMaskMode(width, height, label);
  testVideo.src = video.url;
  emptyVideo.style.display = "none";
  renderVideoOverlay();
}

videoInput1280.addEventListener("change", async () => {
  await uploadTestVideo(videoInput1280.files[0], RAW_W, RAW_H, "1280x720");
  videoInput1280.value = "";
});

videoInput306.addEventListener("change", async () => {
  await uploadTestVideo(videoInput306.files[0], SCALED_W, SCALED_H, "306x306");
  videoInput306.value = "";
});

window.addEventListener("paste", async (event) => {
  const files = [...event.clipboardData.files].filter((file) => file.type.startsWith("image/"));
  for (const file of files) {
    await uploadPastedImage(file);
  }
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("active");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("active");
});

dropZone.addEventListener("drop", async (event) => {
  event.preventDefault();
  dropZone.classList.remove("active");
  const files = [...event.dataTransfer.files].filter((file) => file.type.startsWith("image/"));
  for (const file of files) {
    await uploadImageFile(file);
  }
});

frameImage.addEventListener("load", redraw);

loadState();
