"""Local Gradio UI for PDF / ZIP + local LLM."""
import tempfile
from pathlib import Path

import gradio as gr

from . import llm
from .pdf_utils import extract_text_from_pdf
from .rag import build_context
from .zip_utils import cleanup_dir, create_results_zip, extract_supported_files, list_zip_contents


def _upload_path(file_obj) -> Path:
    """Gradio versions return either a path string or a file-like wrapper."""
    raw_path = getattr(file_obj, "name", None) or file_obj
    return Path(raw_path)


def process_upload(file_obj, question: str, model_path: str,
                   max_tokens: int, temperature: float):
    if file_obj is None:
        return "Please upload a PDF or ZIP file.", None
    try:
        path = _upload_path(file_obj)
        if not path.is_file():
            return "Uploaded file is unavailable.", None
        suffix = path.suffix.lower()
        if suffix not in {".pdf", ".zip"}:
            return "Unsupported file type. Upload .pdf or .zip.", None
        if not model_path or not Path(model_path).expanduser().is_file():
            return "Model path is missing or invalid. Provide a valid GGUF path.", None
        if not llm.is_loaded() or llm.loaded_model_path() != str(Path(model_path).expanduser().resolve()):
            llm.load_model(model_path, n_ctx=4096)

        results = {}
        extract_dir = None
        try:
            if suffix == ".pdf":
                document = extract_text_from_pdf(path)
                results["extracted_text.txt"] = document
                default_question = "Summarize this document with key points."
                prompt_label = "Document context"
            else:
                contents = list_zip_contents(path)
                results["zip_listing.txt"] = "\n".join(
                    f"- {item['name']} ({item['size']} B)" for item in contents[:100]
                )
                extract_dir = extract_supported_files(path)
                parts = []
                for file_path in extract_dir:
                    if file_path.suffix.lower() == ".pdf":
                        text = extract_text_from_pdf(file_path)
                    else:
                        text = file_path.read_text(encoding="utf-8", errors="replace")
                    parts.append(f"=== {file_path.as_posix()} ===\n{text}")
                document = "\n\n".join(parts)
                results["combined_extracted.txt"] = document
                default_question = "Summarize the contents of these documents."
                prompt_label = "Documents"

            q = (question or "").strip() or default_question
            context = build_context(q, document)
            answer = llm.generate(f"{prompt_label}:\n\n{context}\n\nQuestion: {q}",
                                  max_tokens=int(max_tokens), temperature=float(temperature))
            results["answer.txt"] = answer
            results["question.txt"] = q
            # Unique output per request avoids concurrent users overwriting each other's ZIP.
            with tempfile.NamedTemporaryFile(prefix="llm_results_", suffix=".zip", delete=False) as handle:
                output_path = Path(handle.name)
            create_results_zip(results, output_path)
            return answer, str(output_path)
        finally:
            if extract_dir:
                cleanup_dir(extract_dir)
    except Exception as exc:
        return f"Processing failed ({type(exc).__name__}): {exc}", None


def build_ui():
    with gr.Blocks(title="Standalone LLM – PDF & ZIP") as demo:
        gr.Markdown(
            "# Standalone Local LLM\nUpload a **PDF** or **ZIP** of documents, ask a question "
            "(or leave blank for a summary), and download extracted text plus the answer."
        )
        with gr.Row():
            with gr.Column():
                model_in = gr.Textbox(label="Path to GGUF model", placeholder="/path/to/model.gguf")
                file_in = gr.File(label="Upload PDF or ZIP", file_types=[".pdf", ".zip"], type="filepath")
                question = gr.Textbox(label="Question (optional – leave empty for summary)",
                                      lines=2, placeholder="What are the main conclusions?")
                max_tokens = gr.Slider(64, 2048, value=512, step=64, label="Max tokens")
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                btn = gr.Button("Process", variant="primary")
            with gr.Column():
                answer_out = gr.Textbox(label="Answer / Summary", lines=16)
                zip_out = gr.File(label="Download results ZIP")
        btn.click(fn=process_upload, inputs=[file_in, question, model_in, max_tokens, temperature],
                  outputs=[answer_out, zip_out])
    return demo


def main():
    build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
