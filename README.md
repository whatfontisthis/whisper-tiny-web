# Whisper STT (Browser, Local Models)

Offline speech-to-text in the browser using [Transformers.js](https://huggingface.co/docs/transformers.js) + local ONNX models.

## Setup

```bash
git clone https://github.com/whatfontisthis/whisper-tiny-web.git
cd whisper-tiny-web
./download-models.sh      # ~561MB, fetches 3 models from Hugging Face
python3 -m http.server 8000
```

Open `http://localhost:8000`. Pick model → Load Model → Start Recording → Stop → transcription appears.

Model weights are not committed to git (too large). The script downloads them from Hugging Face into [models/](models/). Edit [download-models.sh](download-models.sh) to add or remove models.

## Current Models

Downloaded to [models/](models/). ~561MB total.

| Key | Repo | Size | Notes |
|-----|------|------|-------|
| `tiny-en` | `Xenova/whisper-tiny.en` | 40MB | Fast, English only |
| `small-en` | `Xenova/whisper-small.en` | 237MB | Accurate, English only |
| `small-ko` | `Xenova/whisper-small` + `language: "korean"` | 237MB | Multilingual, set to Korean |

## Other Models You Can Add

Edit [download-models.sh](download-models.sh) and [app.js](app.js) `MODELS` map.

### Size variants (OpenAI Whisper)

| Repo | Params | Size (quantized) | Speed | Accuracy |
|------|--------|------------------|-------|----------|
| `Xenova/whisper-tiny` | 39M | ~40MB | fastest | lowest |
| `Xenova/whisper-tiny.en` | 39M | ~40MB | fastest | English only |
| `Xenova/whisper-base` | 74M | ~75MB | fast | better |
| `Xenova/whisper-base.en` | 74M | ~75MB | fast | English only |
| `Xenova/whisper-small` | 244M | ~237MB | medium | good |
| `Xenova/whisper-small.en` | 244M | ~237MB | medium | English only |
| `Xenova/whisper-medium` | 769M | ~720MB | slow | high |
| `Xenova/whisper-large-v3-turbo` | 809M | ~800MB | medium | best balance |

`.en` variants: English-only, more accurate at same size (model capacity not split across languages).

### Multilingual usage

Pass language flag at inference:

```js
const result = await transcriber(audio, { language: "korean", task: "transcribe" });
// or translate to English:
const result = await transcriber(audio, { language: "korean", task: "translate" });
```

Supported languages (99): english, chinese, german, spanish, russian, korean, french, japanese, portuguese, turkish, polish, catalan, dutch, arabic, swedish, italian, indonesian, hindi, finnish, vietnamese, hebrew, ukrainian, greek, malay, czech, romanian, danish, hungarian, tamil, norwegian, thai, urdu, croatian, bulgarian, lithuanian, latin, maori, malayalam, welsh, slovak, telugu, persian, latvian, bengali, serbian, azerbaijani, slovenian, kannada, estonian, macedonian, breton, basque, icelandic, armenian, nepali, mongolian, bosnian, kazakh, albanian, swahili, galician, marathi, punjabi, sinhala, khmer, shona, yoruba, somali, afrikaans, occitan, georgian, belarusian, tajik, sindhi, gujarati, amharic, yiddish, lao, uzbek, faroese, haitian creole, pashto, turkmen, nynorsk, maltese, sanskrit, luxembourgish, myanmar, tibetan, tagalog, malagasy, assamese, tatar, hawaiian, lingala, hausa, bashkir, javanese, sundanese.

### Non-Whisper ASR alternatives

| Repo | Model | Notes |
|------|-------|-------|
| `Xenova/wav2vec2-base-960h` | Wav2Vec2 | English, Meta, smaller than tiny |
| `Xenova/mms-1b-all-onnx` | MMS | 1000+ languages, Meta |
| `Xenova/moonshine-tiny-ONNX` | Moonshine | English, faster than whisper-tiny |
| `Xenova/moonshine-base-ONNX` | Moonshine | English, faster than whisper-base |

## How to add a model

1. Add to [download-models.sh](download-models.sh):
   ```bash
   download_model "Xenova/whisper-base.en"
   ```
2. Re-run: `./download-models.sh`
3. Add to [app.js](app.js) `MODELS`:
   ```js
   "base-en": { repo: "Xenova/whisper-base.en", opts: {} }
   ```
4. Add `<option>` to [index.html](index.html).

## How it works

See [../FLOW.md](../FLOW.md) for drowsy detector. Whisper pipeline:

```
mic → MediaRecorder (webm/opus)
  → AudioContext decode at 16kHz mono
  → Float32Array samples
  → transcriber(audio, opts)
    → log-mel spectrogram (80 × 3000)
    → Encoder (conv + transformer) → audio embeddings
    → Decoder (transformer, autoregressive) → token IDs
    → Tokenizer detokenize → text
```

Model = ONNX files loaded via `onnxruntime-web` (WASM backend). Tokenizer = JSON file with BPE vocab + merge rules.

## Files per model

Each model dir needs:
- `config.json` — architecture
- `tokenizer.json` — BPE vocab
- `tokenizer_config.json` — tokenizer class + special tokens
- `preprocessor_config.json` — mel filterbank settings
- `generation_config.json` — decoding defaults (max len, suppress tokens)
- `onnx/encoder_model_quantized.onnx` — encoder weights (int8 quantized)
- `onnx/decoder_model_merged_quantized.onnx` — decoder weights (int8 quantized, KV-cache merged)

Non-quantized (`*_fp16.onnx` or no suffix) = ~4x larger, marginal accuracy gain.

## Config in app.js

```js
env.allowLocalModels = true;        // load from disk
env.allowRemoteModels = false;      // no HF hub fetch
env.localModelPath = "./models/";   // resolve <repo> → ./models/<repo>/
```

Transformers.js resolves `Xenova/whisper-tiny.en` → `./models/Xenova/whisper-tiny.en/`.

## Fine-Tuning for Custom Vocab

Teach Whisper new terms (e.g. "Claude Code", "vibe coding") it currently mishears.

### Try first (no training)

Pass `prompt` to bias decoder:
```js
await transcriber(audio, { prompt: "Claude Code, vibe coding, MCP, Anthropic" });
```
Fixes ~80% of cases. Free, instant.

### Fine-tune path (if prompt not enough)

**Dataset:** 150 clips, ~45 min recording
- 50 clips containing target terms (25 sentences × 2 reads, varied tone/speed)
- 50 other tech term clips
- 50 general speech (no target) — prevents catastrophic forgetting

Format: 16kHz mono WAV + `metadata.csv` (`file_name,sentence`).

**Shortcut:** record 20 real clips, clone voice with XTTS-v2 on 5090, generate 500 synthetic. Mix.

**Training (5090, Linux/WSL, CUDA 12.4+):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate peft bitsandbytes librosa jiwer
```

Use LoRA adapter (small output ~30MB, fast):
```python
from peft import LoraConfig, get_peft_model
config = LoraConfig(r=32, target_modules=["q_proj","v_proj"])
model = get_peft_model(base_model, config)
```

whisper-small LoRA on 1000 samples ≈ 15-30 min on 5090. Only cost = electricity.

**Deploy:** export adapter-merged model to ONNX with `optimum-cli`, quantize, drop into `models/your-model/`, add entry to `MODELS` map in [app.js](app.js).

### VRAM needs on 5090 (32GB)

| Model | Full FT | LoRA |
|-------|---------|------|
| whisper-small | 12GB | 4GB |
| whisper-medium | 32GB tight | 10GB |
| whisper-large-v3 | OOM | 16GB |

### Validation

Track WER on held-out set containing target terms. Stop when WER plateau.
