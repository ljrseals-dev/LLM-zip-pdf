"""Simple Gradio web UI for PDF / ZIP + local LLM."""

from pathlib import Path
import tempfile
import gradio as gr

from .pdf_utils import extract_text_from_pdf
from .zip_utils import list_zip_contents, extract_supported_files, create_results_zip, cleanup_dir
from .rag import build_context
from . import llm


def process_upload(
    file_obj,
    question: str,
    model_path: str,
    max_tokens: int,
    temperature: float,
):
    if file_obj is None:
        return "Please upload a PDF or ZIP file.", None

    path = Path(file_obj.name)
    suffix = path.suffix.lower()

    if not model_path or not Path(model_path).exists():
        return "Model path is missing or invalid. Provide a valid GGUF path.", None

    try:
        if not llm.is_loaded():
            llm.load_model(model_path, n_ctx=4096)
    except Exception as e:
        return f"Failed to load model: {e}", None

    results = {}
    answer = ""

    try:
        if suffix == ".pdf":
            text = extract_text_from_pdf(path)
            results["extracted_text.txt"] = text
            q = question.strip() or "Summarize this document with key points."
            context = build_context(q, text)
            prompt = f"Document context:\n\n{context}\n\nQuestion: {q}"
            answer = llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            results["answer.txt"] = answer
            results["question.txt"] = q

        elif suffix == ".zip":
            contents = list_zip_contents(path)
            listing = "\n".join(f"- {c['name']} ({c['size']} B)" for c in contents[:30])
            extract_dir = extract_supported_files(path)
            parts = []
            for f in extract_dir:
                if f.suffix.lower() == ".pdf":
                    try:
                        parts.append(f"=== {f.name} ===\n{extract_text_from_pdf(f)}")
                    except Exception as e:
                        parts.append(f"=== {f.name} (error: {e}) ===")
                else:
                    try:
                        parts.append(f"=== {f.name} ===\n{f.read_text(encoding='utf-8', errors='replace')}")
                    except Exception as e:
                        parts.append(f"=== {f.name} (error: {e}) ===")
            combined = "\n\n".join(parts)
            results["zip_listing.txt"] = listing
            results["combined_extracted.txt"] = combined
            q = question.strip() or "Summarize the contents of these documents."
            context = build_context(q, combined)
            prompt = f"Documents:\n\n{context}\n\nQuestion: {q}"
            answer = llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            results["answer.txt"] = answer
            results["question.txt"] = q
            cleanup_dir(extract_dir)
        else:
            return "Unsupported file type. Upload .pdf or .zip.", None

        # Create downloadable ZIP
        out_path = Path(tempfile.gettempdir()) / "llm_results.zip"
        create_results_zip(results, out_path)
        return answer, str(out_path)

    except Exception as e:
        return f"Error during processing: {e}", None


def build_ui():
    with gr.Blocks(title="Standalone LLM – PDF & ZIP") as demo:
        gr.Markdown(
            """
            # Standalone Local LLM  
            Upload a **PDF** or **ZIP** of documents, ask a question (or leave blank for summary),  
            and download the extracted text + answer as a ZIP.
            """
        )
        with gr.Row():
            with gr.Column():
                model_in = gr.Textbox(
                    label="Path to GGUF model",
                    placeholder="/path/to/model.gguf",
                    value="",
                )
                file_in = gr.File(label="Upload PDF or ZIP", file_types=[".pdf", ".zip"])
                question = gr.Textbox(
                    label="Question (optional – leave empty for summary)",
                    lines=2,
                    placeholder="What are the main conclusions?",
                )
                max_tokens = gr.Slider(64, 2048, value=512, step=64, label="Max tokens")
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                btn = gr.Button("Process", variant="primary")
            with gr.Column():
                answer_out = gr.Textbox(label="Answer / Summary", lines=16)
                zip_out = gr.File(label="Download results ZIP")

        btn.click(
            fn=process_upload,
            inputs=[file_in, question, model_in, max_tokens, temperature],
            outputs=[answer_out, zip_out],
        )
    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
