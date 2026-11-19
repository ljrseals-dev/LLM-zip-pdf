# Sample Usage

## 1. Interactive chat

```bash
python -m src.cli chat --model models/phi-3.5-mini-q4.gguf --threads 4
```

## 2. Summarize a PDF and get a results ZIP

```bash
python -m src.cli process-pdf report.pdf \
  --model models/phi-3.5-mini-q4.gguf \
  --question "What are the three main findings?" \
  --output findings.zip
```

## 3. Process a ZIP of documents

```bash
python -m src.cli process-zip docs_archive.zip \
  --model models/qwen2.5-1.5b-q4.gguf \
  --summarize \
  --output analysis.zip
```

## 4. Web UI

```bash
python -m src.app
# Open http://127.0.0.1:7860
# Enter model path, upload PDF or ZIP, ask a question, download results ZIP
```

## 5. Extract text only (no model needed)

```bash
python -m src.cli process-pdf report.pdf --extract-only --output report.txt
```
