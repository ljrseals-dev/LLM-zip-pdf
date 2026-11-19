# Runtime integration validation

The normal regression suite uses controlled model doubles so it can run quickly without downloading weights. This separate integration check exercises the real `llama-cpp-python` runtime and a real GGUF model, then starts the Gradio application and uploads a ZIP through the Gradio client.

## Automated Linux validation

The `.github/workflows/runtime-integration.yml` workflow downloads the 398 MB Q4_K_M quantization of Qwen2.5-0.5B-Instruct from the public Hugging Face repository `bartowski/Qwen2.5-0.5B-Instruct-GGUF`. The workflow verifies the downloaded model against its published SHA-256 before loading it. The model is Apache-2.0 licensed; the model file is downloaded at test time and is not committed to this repository.

The workflow checks three separate outcomes:

1. The actual GGUF model loads through `llama-cpp-python` and generates a non-empty response.
2. The actual Gradio server starts on localhost and accepts a ZIP upload through `gradio_client`.
3. The result ZIP contains the uploaded source text and the model-generated answer.

This is a small-model CPU smoke test, not a benchmark or a claim that every larger model, hardware backend, or context size works.

## Run the same test locally or on Android Termux

Install the project dependencies and `requests`, then download the same model file from the Hugging Face model repository. Set the model path and run:

```bash
export LLM_ZIP_PDF_MODEL="$HOME/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
python scripts/runtime_smoke.py
```

On Termux, first confirm that `llama-cpp-python` successfully builds or installs for the device's Android/ARM environment. The same smoke script then checks both inference and the local Gradio upload path. It must be run on the actual phone to count as device validation; a Linux CI pass does not establish Android compatibility.

## Boundaries

- The automated workflow tests Linux x86-64 CPU inference, not Android ARM.
- The workflow needs network access to download the model and Python packages.
- It does not upload the model to Git or GitHub Actions artifacts.
- The test exercises ZIP upload through Gradio. Direct PDF upload remains covered by the existing PDF unit/integration tests, but this smoke test does not replace a separate browser-driven PDF upload test.
