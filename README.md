# Standalone Local LLM with ZIP & PDF Support

A lightweight, offline-capable package for running a **standalone language model** that supports:

- **PDF upload** and text extraction / RAG-style Q&A
- **ZIP upload** (extract, process contained documents, or treat as archive of context)
- **ZIP download** of extracted document text, questions, and generated answers

Designed to work on:
- Desktop Linux / macOS / Windows (via Python)
- **Android Termux** (CPU inference, small models recommended)

No cloud or API keys are required after model download. The application binds its web UI to localhost by default; keep model files and generated archives in trusted local storage.

## Features

| Feature | Description |
|---------|-------------|
| PDF handling | Extract text from PDFs (text-based; OCR optional via Tesseract) |
| ZIP handling | Upload ZIP → list/extract contents → process PDFs/TXT/MD inside → re-package results as ZIP |
| Local inference | Uses `llama-cpp-python` with local GGUF models |
| CLI + simple web UI | Command-line for scripting; Gradio UI for interactive use |
| Offline after setup | Once models and deps are installed, runs fully offline |
| Termux friendly | Minimal deps, small model recommendations, wake-lock notes |

## Quick Start

### 1. Requirements

- Python 3.10+
- ~4–8 GB free RAM (depends on model)
- For Termux: install from F-Droid (recommended)

### 2. Install (Desktop)

```bash
git clone https://github.com/ljrseals-dev/LLM-zip-pdf.git
cd LLM-zip-pdf
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install (Termux)

```bash
pkg update && pkg upgrade
pkg install python git clang cmake libandroid-spawn
pip install -r requirements.txt
# Optional: pkg install tesseract  (for scanned PDF OCR)
```

### 4. Get a Model (GGUF)

Download a small quantized GGUF, e.g.:

- Phi-3.5 Mini / Gemma-2-2B / Qwen2.5-1.5B (Q4_K_M recommended for phones)

Place it in `models/` or pass path via CLI.

Example (Hugging Face):

```bash
mkdir -p models
# Use wget / curl or termux download managers
```

### 5. Run

**CLI example:**

```bash
python -m src.cli chat --model models/your-model.gguf
python -m src.cli process-pdf document.pdf --question "Summarize the key points"
python -m src.cli process-zip archive.zip --output results.zip
```

**Web UI (Gradio):**

```bash
python -m src.app
# Open the local URL shown (usually http://127.0.0.1:7860)
```

In the UI:
1. Upload PDF or ZIP
2. Ask questions or request summary / extraction
3. Download results as ZIP

## Project Structure

```
standalone-llm-docs/
├── README.md
├── requirements.txt
├── LICENSE
├── src/
│   ├── __init__.py
│   ├── cli.py          # Command-line interface
│   ├── app.py          # Gradio web UI
│   ├── llm.py          # Model loading & inference
│   ├── pdf_utils.py    # PDF text extraction
│   ├── zip_utils.py    # ZIP extract / create
│   └── rag.py          # Simple chunking + retrieval
├── models/             # Place GGUF files here (gitignored)
├── examples/
│   └── sample_usage.md
└── docs/
    └── termux.md
```

## Termux Notes

- Prefer 1.5B–3B parameter models quantized to Q4/Q5.
- Run `termux-wake-lock` before long inference sessions.
- Keep Termux in foreground; close heavy apps.
- Expect lower tokens/sec than desktop (CPU only unless Vulkan setup is done).
- Storage: models go under `~/storage/shared/` or internal Termux home.

See `docs/termux.md` for detailed setup and performance tips.

## Supported File Types (inside ZIP or direct)

- `.pdf` (text extraction)
- `.txt`, `.md`, `.csv` (plain text)
- Nested ZIPs are listed but not recursively extracted (not supported yet)

## Security and resource limits

ZIP extraction rejects path traversal, symbolic links and other non-regular entries, refuses encrypted members, preserves directory paths to avoid basename collisions, and enforces limits of 10,000 archive entries, 50 supported files, 25 MiB per file, 100 MiB total uncompressed data, and a 200:1 compression ratio. Extraction also refuses to write through symlinks already present in the destination directory. Extraction failures are reported rather than silently continuing. Uploaded documents are processed locally; treat document contents as untrusted input and review generated answers.

## Limitations

- Scanned/image-only PDFs need a separately implemented OCR pipeline; installing Tesseract alone does not enable OCR in this application.
- Very large ZIPs or multi-GB PDFs may hit memory limits on mobile.
- No multi-modal vision models in the base package (text-focused).
- Inference quality depends entirely on the chosen GGUF model.

## License

MIT (see LICENSE). Model weights have their own licenses — check the model card.

## Credits & Inspiration

Built for offline, private document workflows. Compatible with the broader llama.cpp / GGUF ecosystem and Termux local-LLM community projects.

---

**Ready to use.** Clone the repository, install dependencies, add a GGUF model, and start chatting with your PDFs and ZIP archives offline.
