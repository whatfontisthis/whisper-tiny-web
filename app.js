import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";

const isLocalDev = location.hostname === "localhost" || location.hostname === "127.0.0.1";
env.allowLocalModels = isLocalDev;
env.allowRemoteModels = !isLocalDev;
env.localModelPath = "./models/";
env.backends.onnx.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2/dist/";

const recBtn = document.getElementById("recBtn");
const loadBtn = document.getElementById("loadBtn");
const modelSel = document.getElementById("modelSel");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");

const MODELS = {
  "tiny-en":  { repo: "Xenova/whisper-tiny.en",  opts: {} },
  "small-en": { repo: "Xenova/whisper-small.en", opts: {} },
  "small-ko": { repo: "Xenova/whisper-small",    opts: { language: "korean", task: "transcribe" } }
};

let transcriber = null;
let currentOpts = {};
let mediaRecorder = null;
let audioChunks = [];
let recording = false;

async function loadModel() {
  const key = modelSel.value;
  const cfg = MODELS[key];
  loadBtn.disabled = true;
  recBtn.disabled = true;
  statusEl.textContent = `Loading ${cfg.repo}...`;

  transcriber = await pipeline("automatic-speech-recognition", cfg.repo, {
    progress_callback: (p) => {
      if (p.status === "progress") {
        statusEl.textContent = `${p.file}: ${Math.round(p.progress)}%`;
      }
    }
  });
  currentOpts = cfg.opts;
  statusEl.textContent = `Ready: ${cfg.repo}`;
  loadBtn.disabled = false;
  recBtn.disabled = false;
}

async function startRec() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];
  mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
  mediaRecorder.onstop = handleStop;
  mediaRecorder.start();
  recording = true;
  recBtn.textContent = "Stop Recording";
  recBtn.classList.add("recording");
  statusEl.textContent = "Recording...";
}

async function stopRec() {
  mediaRecorder.stop();
  mediaRecorder.stream.getTracks().forEach(t => t.stop());
  recording = false;
  recBtn.classList.remove("recording");
  recBtn.disabled = true;
  recBtn.textContent = "Transcribing...";
  statusEl.textContent = "Processing audio...";
}

async function handleStop() {
  const blob = new Blob(audioChunks, { type: "audio/webm" });
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const decoded = await audioCtx.decodeAudioData(arrayBuffer);
  const audio = decoded.getChannelData(0);

  const result = await transcriber(audio, currentOpts);
  outputEl.textContent = result.text.trim() || "(no speech detected)";
  statusEl.textContent = "Done.";
  recBtn.disabled = false;
  recBtn.textContent = "Start Recording";
}

recBtn.addEventListener("click", () => {
  if (recording) stopRec(); else startRec();
});

loadBtn.addEventListener("click", loadModel);
