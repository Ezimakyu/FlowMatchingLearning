from __future__ import annotations

import io
from functools import lru_cache

import modal

from backend.app.models import VisionExtractionResult, VisionPageExtraction

app = modal.App("phase-a-vision-extraction")

vision_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1")
    .uv_pip_install(
        "pymupdf==1.26.4",
        "pillow==11.3.0",
        "transformers==4.56.2",
        "torch==2.8.0",
    )
)


@lru_cache(maxsize=2)
def _load_caption_pipeline(model_name: str):
    from transformers import pipeline

    try:
        return pipeline("image-to-text", model=model_name, device=0)
    except Exception:
        # Fallback that is much lighter and has broad compatibility.
        return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0)


def _describe_image(image_bytes: bytes, model_name: str) -> str:
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    captioner = _load_caption_pipeline(model_name)
    result = captioner(image, max_new_tokens=64)
    if not result:
        return ""
    generated = result[0].get("generated_text", "").strip()
    return generated


@app.function(
    image=vision_image,
    gpu="T4",
    timeout=60 * 20,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=2.0),
)
def extract_document_vision(
    *,
    file_bytes: bytes,
    doc_id: str,
    source_file_id: str,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
) -> dict:
    import fitz

    document = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[VisionPageExtraction] = []

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        raw_text = page.get_text("text").strip()

        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image_bytes = pixmap.tobytes("png")
        description = _describe_image(image_bytes=image_bytes, model_name=model_name)

        pages.append(
            VisionPageExtraction(
                page_number=page_index + 1,
                raw_text=raw_text,
                image_descriptions=[description] if description else [],
                chunk_ids=[],
            )
        )

    result = VisionExtractionResult(
        doc_id=doc_id,
        source_file_id=source_file_id,
        model_name=model_name,
        pages=pages,
        metadata={"page_count": len(pages)},
    )
    return result.model_dump(mode="json")


@app.local_entrypoint()
def main(input_path: str, doc_id: str, source_file_id: str) -> None:
    with open(input_path, "rb") as file_handle:
        file_bytes = file_handle.read()
    payload = extract_document_vision.remote(
        file_bytes=file_bytes,
        doc_id=doc_id,
        source_file_id=source_file_id,
    )
    print(payload)
