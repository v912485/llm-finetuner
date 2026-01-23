import logging
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("training")


def _resolve_converter_path() -> Path | None:
    converter_path = os.getenv("GGUF_CONVERTER_PATH")
    if converter_path:
        path = Path(converter_path)
        return path if path.is_file() else None

    llama_cpp_dir = os.getenv("LLAMA_CPP_DIR")
    if not llama_cpp_dir:
        return None

    path = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    return path if path.is_file() else None


def _is_adapter_only(model_dir: Path) -> bool:
    return (model_dir / "adapter_config.json").exists()


def _merge_adapter_model(adapter_dir: Path, base_model_id: str, output_dir: Path) -> None:
    logger.info("Merging adapter weights into base model for GGUF conversion")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = peft_model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def convert_model_to_gguf(model_dir: Path, output_path: Path, base_model_id: str | None) -> None:
    converter_path = _resolve_converter_path()
    if not converter_path:
        raise RuntimeError(
            "GGUF conversion requires LLAMA_CPP_DIR or GGUF_CONVERTER_PATH to be set"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    outtype = os.getenv("GGUF_OUTTYPE", "f16")

    if _is_adapter_only(model_dir):
        if not base_model_id:
            raise RuntimeError("Base model ID is required to merge LoRA adapters for GGUF conversion")
        with TemporaryDirectory() as tmp_dir:
            merged_dir = Path(tmp_dir) / "merged"
            merged_dir.mkdir(parents=True, exist_ok=True)
            _merge_adapter_model(model_dir, base_model_id, merged_dir)
            _run_converter(converter_path, merged_dir, output_path, outtype)
        return

    _run_converter(converter_path, model_dir, output_path, outtype)


def _run_converter(converter_path: Path, model_dir: Path, output_path: Path, outtype: str) -> None:
    logger.info(f"Converting model to GGUF using {converter_path}")
    command = [
        sys.executable,
        str(converter_path),
        str(model_dir),
        "--outfile",
        str(output_path),
        "--outtype",
        outtype
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        message = stderr or stdout or "Unknown conversion failure"
        raise RuntimeError(f"GGUF conversion failed: {message}")
