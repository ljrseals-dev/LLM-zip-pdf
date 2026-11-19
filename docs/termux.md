# Termux Setup Guide

Running the standalone LLM package on Android via Termux.

## Prerequisites

1. Install **Termux** from F-Droid (recommended over Play Store).
2. Optionally install Termux:API and Termux:Widget.
3. Grant storage permission:
   ```bash
   termux-setup-storage
   ```

## Installation Steps

```bash
pkg update && pkg upgrade -y
pkg install python git clang cmake wget libandroid-spawn -y

# Optional OCR support for scanned PDFs
# pkg install tesseract poppler -y

cd ~
# Unzip the package or clone
# unzip standalone-llm-docs.zip
cd standalone-llm-docs

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on llama-cpp-python**: Building from source on Termux can take time and may require extra flags.  
> Prefer a small model and pure CPU first. Some users share prebuilt wheels in the community.

## Recommended Models (Phone)

| Model              | Approx size | RAM needed | Notes                     |
|--------------------|-------------|------------|---------------------------|
| Qwen2.5-1.5B Q4_K_M| ~1.1 GB    | ~2–3 GB   | Good multilingual balance |
| Gemma-2-2B Q4      | ~1.5 GB    | ~3–4 GB   | Strong general quality    |
| Phi-3.5-mini Q4    | ~2.2 GB    | ~4 GB     | Solid reasoning           |
| TinyLlama / SmolLM | <1 GB      | ~1.5 GB   | Fastest / lowest RAM      |

Download GGUF files from Hugging Face and place them under `~/models/` or `/sdcard/AI_Models/`.

## Running

```bash
# Keep the process alive
termux-wake-lock

source .venv/bin/activate
python -m src.cli chat --model ~/models/your-model.gguf --threads 4

# Or start the Gradio UI (open in phone browser)
python -m src.app
# Then visit http://127.0.0.1:7860
```

## Performance Tips

- Close background apps and browsers.
- Plug in charger; inference is power-hungry.
- Use 2–4 threads; more may overheat or throttle.
- Prefer Q4/Q5 quantizations.
- For longer sessions keep Termux in the foreground.
- `termux-wake-lock` prevents Android from sleeping the process.

## Storage Locations

- Internal Termux home: `~/`
- Shared storage: `~/storage/shared/` or `/sdcard/`
- Models can live on SD card if internal space is tight.

## Troubleshooting

- **Out of memory**: Switch to a smaller model or lower context (`--ctx 2048`).
- **Slow first token**: Normal on first load; subsequent prompts are faster.
- **Build failures for llama-cpp-python**: Search Termux GitHub issues or community scripts for prebuilt binaries / specific CMAKE flags.
- **Permission denied on files**: Ensure `termux-setup-storage` was run and the file is under accessible paths.

Happy offline document chatting on your phone!
