from flask import Blueprint, jsonify, request
from models.model_manager import ModelManager
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import MODELS_DIR, SAVED_MODELS_DIR
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
import logging
import torch
import json
from datetime import datetime
from pathlib import Path
import os
from urllib.parse import urlparse
import hashlib
import shutil
from huggingface_hub import model_info
import requests
try:
    import sglang as sgl
except Exception:
    sgl = None

bp = Blueprint('models', __name__, url_prefix='/api/models')
logger = logging.getLogger('training')
model_manager = ModelManager()

# Global variable to hold the SGLang runtime (to avoid reloading)
sgl_runtime_cache = {}

def _parse_float(data, key: str, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    value = data.get(key, default)
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        value_f = float(default)
    if min_value is not None and value_f < min_value:
        return float(min_value)
    if max_value is not None and value_f > max_value:
        return float(max_value)
    return value_f


def _parse_int(data, key: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    value = data.get(key, default)
    try:
        value_i = int(value)
    except (TypeError, ValueError):
        value_i = int(default)
    if min_value is not None and value_i < min_value:
        return int(min_value)
    if max_value is not None and value_i > max_value:
        return int(max_value)
    return value_i

def _get_context_limit(tokenizer, model):
    """Return best-effort context limit for the model, or None if unknown."""
    limits = []
    tok_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_limit, int) and 0 < tok_limit < 10**6:
        limits.append(tok_limit)
    cfg = getattr(model, "config", None)
    cfg_limit = getattr(cfg, "max_position_embeddings", None)
    if isinstance(cfg_limit, int) and cfg_limit > 0:
        limits.append(cfg_limit)
    return min(limits) if limits else None

def _clamp_max_new_tokens(tokenizer, model, input_ids_len: int, requested_max_new_tokens: int) -> int:
    context_limit = _get_context_limit(tokenizer, model)
    if not context_limit:
        return requested_max_new_tokens
    available = max(1, context_limit - int(input_ids_len))
    if requested_max_new_tokens <= available:
        return requested_max_new_tokens
    logger.warning(
        f"Clamping max_new_tokens from {requested_max_new_tokens} to {available} "
        f"to fit within context_limit={context_limit} (input_len={int(input_ids_len)})."
    )
    return available

def _sanitize_saved_model_name(name: str) -> str:
    safe = "".join(c for c in (name or "") if c.isalnum() or c in ("-", "_")).strip()
    if not safe:
        raise ValueError("Invalid saved model name")
    return safe


def _load_app_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def _get_ollama_base_url() -> str:
    env_host = os.getenv("OLLAMA_HOST")
    env_port = os.getenv("OLLAMA_PORT")
    config = _load_app_config()
    ollama_cfg = config.get("ollama", {}) if isinstance(config, dict) else {}
    host = (env_host or ollama_cfg.get("host") or "http://localhost").strip()
    port = env_port or ollama_cfg.get("port") or 11434

    parsed = urlparse(host if "://" in host else f"http://{host}")
    netloc = parsed.netloc or parsed.path
    scheme = parsed.scheme or "http"
    hostname = parsed.hostname or netloc
    port_in_host = parsed.port is not None
    if not hostname:
        hostname = "localhost"
    if port_in_host:
        base = f"{scheme}://{hostname}:{parsed.port}"
    else:
        base = f"{scheme}://{hostname}:{int(port)}"
    return base.rstrip("/")


def _find_gguf_files(model_dir: Path) -> list[Path]:
    return [path for path in model_dir.rglob("*.gguf") if path.is_file()]


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _upload_ollama_blob(ollama_base: str, gguf_path: Path) -> str:
    digest = _sha256_file(gguf_path)
    blob_ref = f"sha256:{digest}"
    blob_url = f"{ollama_base}/api/blobs/{blob_ref}"
    try:
        head = requests.head(blob_url, timeout=30)
    except requests.RequestException as req_err:
        raise RuntimeError(f"Ollama blob check failed: {req_err}") from req_err

    if head.status_code == 404:
        try:
            with open(gguf_path, "rb") as f:
                upload = requests.post(
                    blob_url,
                    data=f,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=600
                )
        except requests.RequestException as req_err:
            raise RuntimeError(f"Ollama blob upload failed: {req_err}") from req_err
        if not upload.ok:
            raise RuntimeError(f"Ollama blob upload failed ({upload.status_code})")
    elif not head.ok:
        raise RuntimeError(f"Ollama blob check failed ({head.status_code})")

    return blob_ref


def _build_saved_model_entry(model_dir: Path) -> dict:
    metadata = {}
    metadata_path = model_dir / 'metadata.json'
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata for {model_dir}: {str(e)}")
            metadata = {}

    saved_date = metadata.get('saved_date', '')
    if not saved_date:
        saved_date = metadata.get('save_date', '')

    return {
        'name': model_dir.name,
        'original_model': metadata.get('original_model', ''),
        'saved_date': saved_date,
        'description': metadata.get('description', '')
    }


def _resolve_base_model_path(model_path: Path) -> tuple[Path, bool]:
    is_peft = (model_path / "adapter_config.json").exists()
    if not is_peft:
        return model_path, False

    logger.info(f"Loading LoRA adapter from {model_path}")
    base_model_id = None
    metadata_path = model_path / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                base_model_id = metadata.get('original_model') or metadata.get('base_model_id')
        except Exception as e:
            logger.warning(f"Error reading metadata.json: {e}")

    if not base_model_id:
        config_path = model_path / "training_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    t_config = json.load(f)
                    base_model_id = t_config.get('model_id')
            except Exception as e:
                logger.warning(f"Error reading training_config.json: {e}")

    if not base_model_id:
        raise ValueError(f"Could not determine base model for adapter at {model_path}")

    logger.info(f"Base model identified: {base_model_id}")
    safe_base_name = base_model_id.replace('/', '_')
    base_model_path = MODELS_DIR / safe_base_name
    if not base_model_path.exists():
        raise ValueError(f"Base model {base_model_id} not found at {base_model_path}. Please download it first.")

    return base_model_path, True


def _load_model_and_tokenizer(model_path: Path, device_type: str):
    """Helper to load model and tokenizer, handling both base models and LoRA adapters."""
    base_model_path, is_peft = _resolve_base_model_path(model_path)
    if is_peft:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            dtype=torch.float16 if device_type in ('cuda', 'rocm') else torch.float32,
        )
        if PeftModel is None:
            raise ValueError("PEFT library not installed; cannot load LoRA adapter.")
        model = PeftModel.from_pretrained(base_model, model_path)
        return model, tokenizer

    logger.info(f"Loading standard model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        dtype=torch.float16 if device_type in ('cuda', 'rocm') else torch.float32,
    )
    return model, tokenizer


def _load_tokenizer(model_path: Path):
    base_model_path, _ = _resolve_base_model_path(model_path)
    return AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)


def _build_chat_prompt(messages: list[dict], tokenizer):
    add_generation_prompt = True
    if messages and messages[-1].get("role") == "assistant":
        add_generation_prompt = False

    if tokenizer is not None:
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception as e:
                logger.warning(f"Chat template apply failed; using fallback format. Details: {e}")

    prompt = ""
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        prompt += f"<|{role}|>\n{content}\n"
    if add_generation_prompt:
        prompt += "<|assistant|>\n"
    return prompt


def get_sgl_runtime(model_path_str: str, max_tokens: int, device_type: str):
    """Helper function to load and cache SGLang runtime."""
    global sgl_runtime_cache
    if sgl is None:
        logger.warning("SGLang is not available; cannot initialise runtime.")
        return None
    try:
        import orjson  # noqa: F401
    except ModuleNotFoundError:
        logger.warning("SGLang dependency missing: 'orjson'. Install it to enable SGLang runtime.")
        return None
    if model_path_str in sgl_runtime_cache:
        logger.info(f"Using cached SGLang runtime for {model_path_str}")
        return sgl_runtime_cache[model_path_str]

    # Skip SGLang for LoRA adapters for now, as it requires special base model handling
    if (Path(model_path_str) / "adapter_config.json").exists():
        logger.info(f"SGLang: Detected LoRA adapter at {model_path_str}. Falling back to Transformers for LoRA support.")
        sgl_runtime_cache[model_path_str] = None
        return None

    logger.info(f"Initializing SGLang runtime for {model_path_str}")
    model_config = {
        "model_path": model_path_str,
        "tensor_parallel_size": 1,  # Adjust if using multiple GPUs
        "max_tokens": max_tokens,   # Use max_tokens from request
        "trust_remote_code": True,
        "dtype": "float16" if device_type in ('cuda', 'rocm') else "float32"
    }

    try:
        # Preflight: if sgl_kernel cannot load (e.g. missing libnvrtc / wrong arch),
        # don't even attempt runtime initialisation (it can produce noisy destructor errors).
        try:
            import sgl_kernel  # noqa: F401
        except Exception as kernel_e:
            logger.warning(f"SGLang kernel unavailable; falling back to Transformers. Details: {kernel_e}")
            sgl_runtime_cache[model_path_str] = None
            return None

        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if device_type == 'rocm' or (device_type == 'cuda' and is_rocm):
            sgl.set_default_backend("rocm")
            logger.info("SGLang: Set default backend to ROCm")
        elif device_type == 'cuda':
            sgl.set_default_backend("cuda")
            logger.info("SGLang: Set default backend to CUDA")
        else:
            sgl.set_default_backend("cpu")
            logger.info("SGLang: Set default backend to CPU")

        runtime = sgl.Runtime(**model_config) # Use ** to unpack config
        sgl_runtime_cache[model_path_str] = runtime
        logger.info(f"SGLang runtime initialized successfully for {model_path_str}. Backend: {sgl.get_default_backend()}")
        return runtime
    except Exception as e:
        logger.error(f"Failed to initialize SGLang runtime for {model_path_str}: {e}", exc_info=True)
        sgl_runtime_cache[model_path_str] = None
        return None

@bp.route('/available', methods=['GET'])
def get_available_models():
    return jsonify(model_manager.get_available_models())

@bp.route('/downloaded', methods=['GET'])
def get_downloaded_models():
    return jsonify(model_manager.get_downloaded_models())

@bp.route('/download', methods=['POST'])
def download_model():
    data = request.json
    model_id = data.get('model_id')
    
    if not model_id:
        return jsonify({
            "status": "error",
            "message": "Model ID is required"
        }), 400
        
    return model_manager.download_model(model_id) 

@bp.route('/inference', methods=['POST'])
def run_inference():
    try:
        data = request.json or {}

        # Backwards-compatible aliases (older frontends)
        if 'input' not in data and 'prompt' in data:
            data['input'] = data.get('prompt')
        if 'max_length' not in data and 'max_tokens' in data:
            data['max_length'] = data.get('max_tokens')
        
        # Get generation parameters
        temperature = _parse_float(data, 'temperature', 0.7, min_value=0.0, max_value=5.0)
        max_length = _parse_int(data, 'max_length', 512, min_value=1, max_value=1000000)
        do_sample = bool(data.get('do_sample', True))
        top_p = _parse_float(data, 'top_p', 0.95, min_value=0.0, max_value=1.0)
        top_k = _parse_int(data, 'top_k', 0, min_value=0, max_value=1000000)
        typical_p = _parse_float(data, 'typical_p', 1.0, min_value=0.0, max_value=1.0)
        repetition_penalty = _parse_float(data, 'repetition_penalty', 1.0, min_value=0.0, max_value=10.0)
        no_repeat_ngram_size = _parse_int(data, 'no_repeat_ngram_size', 0, min_value=0, max_value=1000000)

        logger.info(
            "Generation parameters: "
            f"temp={temperature}, max_length={max_length}, do_sample={do_sample}, "
            f"top_p={top_p}, top_k={top_k}, typical_p={typical_p}, "
            f"repetition_penalty={repetition_penalty}, no_repeat_ngram_size={no_repeat_ngram_size}"
        )
        
        model_id = data.get('model_id')
        input_text = data.get('input')
        use_finetuned = data.get('use_finetuned', False)
        saved_model_name = data.get('saved_model_name')
        
        if not input_text:
            logger.error("Missing input text")
            return jsonify({
                "status": "error",
                "message": "Input text is required"
            }), 400
            
        logger.info(f"Input text (first 100 chars): {input_text[:100]}...")
        
        if not model_id and not saved_model_name:
            logger.error("Neither model_id nor saved_model_name provided")
            return jsonify({
                "status": "error",
                "message": "Either model_id or saved_model_name is required"
            }), 400
            
        # Determine model path
        if saved_model_name:
            try:
                safe_saved_model_name = _sanitize_saved_model_name(saved_model_name)
            except ValueError as ve:
                return jsonify({"status": "error", "message": str(ve)}), 400
            model_path = SAVED_MODELS_DIR / safe_saved_model_name
            logger.info(f"Using saved model: {safe_saved_model_name}")
        else:
            safe_model_name = model_id.replace('/', '_')
            model_path = MODELS_DIR / safe_model_name
            if use_finetuned:
                # Assuming finetuned models are saved within the base model dir
                # Adjust if finetuned models are saved elsewhere
                model_path = model_path / 'finetuned' # Placeholder, adjust path if needed
                logger.info(f"Using fine-tuned model path: {model_path}")
            else:
                 logger.info(f"Using base model path: {model_path}")
                
        if not model_path.exists():
            logger.error(f"Model path not found: {model_path}")
            return jsonify({
                "status": "error",
                "message": f"Model not found at {model_path}"
            }), 404

        model_path_str = str(model_path)

        # Get device info (still useful for logging)
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        device_type = 'rocm' if is_rocm else ('cuda' if torch.cuda.is_available() else 'cpu')
        device_info = {
            'type': device_type,
            'name': torch.cuda.get_device_name(0) if device_type in ('cuda', 'rocm') else 'CPU',
            'memory': f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB" if device_type in ('cuda', 'rocm') else 'N/A'
        }
        logger.info(f"Detected device: {device_info['type']} - {device_info['name']} ({device_info['memory']})")

        # Get SGLang runtime (optional). If unavailable, fall back to Transformers generation.
        sgl_model = get_sgl_runtime(model_path_str, max_length, device_type)
        if sgl_model is None:
            try:
                model, tokenizer = _load_model_and_tokenizer(model_path, device_type)
                model = model.to(torch.device(device_type))
                inputs = tokenizer(input_text, return_tensors="pt").to(torch.device(device_type))
                max_new_tokens = _clamp_max_new_tokens(
                    tokenizer=tokenizer,
                    model=model,
                    input_ids_len=inputs["input_ids"].shape[-1],
                    requested_max_new_tokens=max_length,
                )
                generate_kwargs = {}
                if repetition_penalty != 1.0:
                    generate_kwargs["repetition_penalty"] = float(repetition_penalty)
                if no_repeat_ngram_size:
                    generate_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 0.0,
                        top_p=top_p if do_sample else 1.0,
                        top_k=top_k if do_sample else 0,
                        typical_p=typical_p if do_sample else 1.0,
                        pad_token_id=tokenizer.eos_token_id,
                        **generate_kwargs,
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return jsonify({"status": "success", "response": response, "device_info": device_info})
            except Exception as fallback_e:
                return jsonify({"status": "error", "message": f"Inference unavailable: {fallback_e}"}), 500

        # Log memory before generation (if GPU)
        if device_type in ('cuda', 'rocm'):
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"GPU memory before generation - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        # Create generation state
        if repetition_penalty != 1.0 or no_repeat_ngram_size or top_k or typical_p != 1.0:
            logger.info(
                "SGLang generation: repetition_penalty/no_repeat_ngram_size/top_k/typical_p are not applied "
                "(not supported via RuntimeState here)."
            )
        state = sgl.RuntimeState(
            temperature=temperature if do_sample else 0.0, # Set temp to 0 if not sampling
            top_p=top_p if do_sample else 1.0,
            max_new_tokens=max_length # Use max_length for max_new_tokens
        )
        logger.info(f"SGLang RuntimeState created: temp={state.temperature}, top_p={state.top_p}, max_new_tokens={state.max_new_tokens}")

        # Generate response using SGLang
        try:
            response = sgl_model.generate(prompt=input_text, state=state)
            logger.info("SGLang generation successful.")
        except Exception as sgl_error:
            logger.error(f"SGLang generation error: {sgl_error}", exc_info=True)
            return jsonify({"status": "error", "message": f"SGLang generation failed: {sgl_error}"}), 500

        # Log memory after generation (if GPU)
        if device_type in ('cuda', 'rocm'):
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"GPU memory after generation - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        # Log the model response
        logger.info(f"Model response (first 100 chars): {response[:100]}...")

        return jsonify({
            "status": "success",
            "response": response,
            "device_info": device_info
        })

    except Exception as e:
        logger.error(f"Inference endpoint error: {str(e)}", exc_info=True) # Log traceback
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route('/saved', methods=['GET'])
def get_saved_models():
    try:
        saved_models = []
        
        if SAVED_MODELS_DIR.exists():
            for model_dir in SAVED_MODELS_DIR.iterdir():
                if model_dir.is_dir():
                    saved_models.append(_build_saved_model_entry(model_dir))
        
        logger.info(f"Found {len(saved_models)} saved models")
        return jsonify({
            "status": "success",
            "saved_models": saved_models
        })
        
    except Exception as e:
        logger.error(f"Error getting saved models: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500 


@bp.route('/saved/<saved_model_name>', methods=['DELETE'])
def delete_saved_model(saved_model_name):
    try:
        try:
            safe_saved_model_name = _sanitize_saved_model_name(saved_model_name)
        except ValueError as ve:
            return jsonify({"status": "error", "message": str(ve)}), 400

        saved_model_path = (SAVED_MODELS_DIR / safe_saved_model_name).resolve()
        base_dir = SAVED_MODELS_DIR.resolve()
        if base_dir not in saved_model_path.parents and saved_model_path != base_dir:
            return jsonify({"status": "error", "message": "Invalid saved model path"}), 400

        if not saved_model_path.exists():
            return jsonify({"status": "error", "message": "Saved model not found"}), 404

        if not saved_model_path.is_dir():
            return jsonify({"status": "error", "message": "Saved model path is not a directory"}), 400

        shutil.rmtree(saved_model_path)
        sgl_runtime_cache.pop(str(saved_model_path), None)
        logger.info(f"Deleted saved model files at {saved_model_path}")
        return jsonify({
            "status": "success",
            "message": "Saved model deleted"
        })
    except Exception as e:
        logger.error(f"Error deleting saved model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@bp.route('/saved/<saved_model_name>/rename', methods=['PATCH'])
def rename_saved_model(saved_model_name):
    try:
        data = request.json or {}
        new_name = data.get('name')
        if not new_name:
            return jsonify({"status": "error", "message": "New name is required"}), 400

        try:
            safe_saved_model_name = _sanitize_saved_model_name(saved_model_name)
            safe_new_name = _sanitize_saved_model_name(new_name)
        except ValueError as ve:
            return jsonify({"status": "error", "message": str(ve)}), 400

        if safe_saved_model_name == safe_new_name:
            return jsonify({"status": "error", "message": "New name must be different"}), 400

        saved_model_path = (SAVED_MODELS_DIR / safe_saved_model_name).resolve()
        target_path = (SAVED_MODELS_DIR / safe_new_name).resolve()
        base_dir = SAVED_MODELS_DIR.resolve()
        if base_dir not in saved_model_path.parents and saved_model_path != base_dir:
            return jsonify({"status": "error", "message": "Invalid saved model path"}), 400
        if base_dir not in target_path.parents and target_path != base_dir:
            return jsonify({"status": "error", "message": "Invalid target model path"}), 400

        if not saved_model_path.exists() or not saved_model_path.is_dir():
            return jsonify({"status": "error", "message": "Saved model not found"}), 404
        if target_path.exists():
            return jsonify({"status": "error", "message": "A saved model with that name already exists"}), 409

        saved_model_path.rename(target_path)
        sgl_runtime_cache.pop(str(saved_model_path), None)
        updated_entry = _build_saved_model_entry(target_path)
        return jsonify({
            "status": "success",
            "message": "Saved model renamed",
            "saved_model": updated_entry
        })
    except Exception as e:
        logger.error(f"Error renaming saved model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@bp.route('/ollama/deploy', methods=['POST'])
def deploy_to_ollama():
    try:
        data = request.json or {}
        saved_model_name = data.get("saved_model_name")
        ollama_model_name = (data.get("ollama_model_name") or saved_model_name or "").strip()
        if not saved_model_name:
            return jsonify({"status": "error", "message": "saved_model_name is required"}), 400
        if not ollama_model_name:
            return jsonify({"status": "error", "message": "ollama_model_name is required"}), 400

        try:
            safe_saved_model_name = _sanitize_saved_model_name(saved_model_name)
        except ValueError as ve:
            return jsonify({"status": "error", "message": str(ve)}), 400

        saved_model_path = (SAVED_MODELS_DIR / safe_saved_model_name).resolve()
        base_dir = SAVED_MODELS_DIR.resolve()
        if base_dir not in saved_model_path.parents and saved_model_path != base_dir:
            return jsonify({"status": "error", "message": "Invalid saved model path"}), 400
        if not saved_model_path.exists() or not saved_model_path.is_dir():
            return jsonify({"status": "error", "message": "Saved model not found"}), 404

        gguf_files = _find_gguf_files(saved_model_path)
        if not gguf_files:
            return jsonify({"status": "error", "message": "No GGUF file found in saved model"}), 404
        if len(gguf_files) > 1:
            candidates = [str(path.relative_to(saved_model_path)) for path in gguf_files]
            return jsonify({
                "status": "error",
                "message": "Multiple GGUF files found; please keep only one",
                "candidates": candidates
            }), 409

        gguf_path = gguf_files[0]
        ollama_base = _get_ollama_base_url()
        try:
            blob_ref = _upload_ollama_blob(ollama_base, gguf_path)
        except Exception as upload_err:
            logger.error(f"Ollama blob upload failed: {upload_err}")
            return jsonify({
                "status": "error",
                "message": f"Ollama blob upload failed: {upload_err}"
            }), 502

        payload = {
            "model": ollama_model_name,
            "files": {gguf_path.name: blob_ref},
            "stream": False
        }
        try:
            response = requests.post(
                f"{ollama_base}/api/create",
                json=payload,
                timeout=600
            )
        except requests.RequestException as req_err:
            logger.error(f"Ollama request failed for {ollama_base}: {req_err}")
            return jsonify({
                "status": "error",
                "message": f"Ollama request failed: {req_err}"
            }), 502

        try:
            response_data = response.json()
        except ValueError:
            response_data = {"raw": response.text}

        if not response.ok:
            logger.error(
                "Ollama create failed: "
                f"status={response.status_code}, response={response_data}"
            )
            detail = response_data.get("error") if isinstance(response_data, dict) else response_data
            if detail is None and isinstance(response_data, dict):
                detail = response_data.get("message") or response_data.get("raw")
            modelfile_payload = {
                "model": ollama_model_name,
                "modelfile": f"FROM {gguf_path.as_posix()}",
                "stream": False
            }
            try:
                fallback_response = requests.post(
                    f"{ollama_base}/api/create",
                    json=modelfile_payload,
                    timeout=600
                )
                try:
                    fallback_data = fallback_response.json()
                except ValueError:
                    fallback_data = {"raw": fallback_response.text}

                if fallback_response.ok:
                    return jsonify({
                        "status": "success",
                        "message": "Model deployed to Ollama",
                        "ollama_response": fallback_data
                    })

                logger.error(
                    "Ollama create fallback failed: "
                    f"status={fallback_response.status_code}, response={fallback_data}"
                )
            except requests.RequestException as fallback_err:
                logger.error(f"Ollama modelfile fallback failed: {fallback_err}")

            return jsonify({
                "status": "error",
                "message": f"Ollama create failed ({response.status_code})",
                "details": detail,
                "ollama_response": response_data
            }), 502

        return jsonify({
            "status": "success",
            "message": "Model deployed to Ollama",
            "ollama_response": response_data
        })
    except Exception as e:
        logger.error(f"Error deploying to Ollama: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route('/cancel-training', methods=['POST'])
def cancel_training():
    try:
        model_manager.cancel_training()
        return jsonify({
            "status": "success",
            "message": "Training cancelled"
        })
    except ValueError as ve:
        return jsonify({
            "status": "error",
            "message": str(ve)
        }), 400
    except Exception as e:
        logger.error(f"Error cancelling training: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500 

@bp.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint using SGLang"""
    try:
        data = request.json
        messages = data.get('messages', [])
        if not messages:
            return jsonify({"error": {"message": "messages is required", "type": "invalid_request_error", "code": "invalid_messages"}}), 400

        model_id_openai = data.get('model') # Model name from OpenAI request
        temperature = _parse_float(data, 'temperature', 0.7, min_value=0.0, max_value=5.0)
        max_tokens = _parse_int(data, 'max_tokens', 512, min_value=1, max_value=1000000) # Use max_tokens from request
        top_p = _parse_float(data, 'top_p', 0.95, min_value=0.0, max_value=1.0)
        top_k = _parse_int(data, 'top_k', 0, min_value=0, max_value=1000000)
        typical_p = _parse_float(data, 'typical_p', 1.0, min_value=0.0, max_value=1.0)
        repetition_penalty = _parse_float(data, 'repetition_penalty', 1.0, min_value=0.0, max_value=10.0)
        no_repeat_ngram_size = _parse_int(data, 'no_repeat_ngram_size', 0, min_value=0, max_value=1000000)
        # stream = data.get('stream', False) # SGLang might support streaming differently

        logger.info(
            "Chat parameters: "
            f"temp={temperature}, max_tokens={max_tokens}, top_p={top_p}, top_k={top_k}, typical_p={typical_p}, "
            f"repetition_penalty={repetition_penalty}, no_repeat_ngram_size={no_repeat_ngram_size}"
        )

        # Determine model path (similar logic to /inference)
        model_path = None
        if model_id_openai:
             # Check saved models first
            try:
                safe_saved_name = _sanitize_saved_model_name(model_id_openai)
            except Exception:
                safe_saved_name = None

            saved_model_path = SAVED_MODELS_DIR / safe_saved_name if safe_saved_name else None
            if saved_model_path and saved_model_path.exists() and saved_model_path.is_dir():
                model_path = saved_model_path
                logger.info(f"Using saved model for chat: {model_id_openai}")
            else:
                # Assume it's a base model ID from config
                safe_model_name = model_id_openai.replace('/', '_')
                base_model_path = MODELS_DIR / safe_model_name
                if base_model_path.exists() and base_model_path.is_dir():
                     model_path = base_model_path
                     logger.info(f"Using base model for chat: {model_id_openai}")

        if not model_path:
             logger.error(f"Could not find saved or base model matching: {model_id_openai}")
             # Maybe try finding the first downloaded model as a fallback? Or error out.
             return jsonify({"error": {"message": f"Model '{model_id_openai}' not found.", "type": "invalid_request_error", "code": "model_not_found"}}), 404

        model_path_str = str(model_path)

        # Get device info
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        device_type = 'rocm' if is_rocm else ('cuda' if torch.cuda.is_available() else 'cpu')
        device_info = { # Recreate device_info dict here
            'type': device_type,
            'name': torch.cuda.get_device_name(0) if device_type in ('cuda', 'rocm') else 'CPU',
            'memory': f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB" if device_type in ('cuda', 'rocm') else 'N/A'
        }
        logger.info(f"Chat - Detected device: {device_info['type']} - {device_info['name']} ({device_info['memory']})")


        # Get SGLang runtime (optional); if unavailable, return a clear error for now.
        sgl_model = get_sgl_runtime(model_path_str, max_tokens, device_type) # Pass max_tokens
        if sgl_model is None:
            try:
                model, tokenizer = _load_model_and_tokenizer(model_path, device_type)
                model = model.to(torch.device('cuda' if device_type in ('cuda', 'rocm') else 'cpu'))

                prompt = _build_chat_prompt(messages, tokenizer)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                max_new_tokens = _clamp_max_new_tokens(
                    tokenizer=tokenizer,
                    model=model,
                    input_ids_len=inputs["input_ids"].shape[-1],
                    requested_max_new_tokens=max_tokens,
                )
                do_sample = temperature > 0
                generate_kwargs = {}
                if repetition_penalty != 1.0:
                    generate_kwargs["repetition_penalty"] = float(repetition_penalty)
                if no_repeat_ngram_size:
                    generate_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 0.0,
                        top_p=top_p if do_sample else 1.0,
                        top_k=top_k if do_sample else 0,
                        typical_p=typical_p if do_sample else 1.0,
                        pad_token_id=tokenizer.eos_token_id,
                        **generate_kwargs,
                    )
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                decoded_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                response_text = decoded[len(decoded_prompt):] if decoded.startswith(decoded_prompt) else decoded

                completion_timestamp = int(datetime.now().timestamp())
                prompt_tokens = int(inputs["input_ids"].shape[-1])
                completion_tokens = int(outputs[0].shape[-1] - inputs["input_ids"].shape[-1])
                return jsonify({
                    "id": f"chatcmpl-{completion_timestamp}",
                    "object": "chat.completion",
                    "created": completion_timestamp,
                    "model": model_id_openai,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text.strip()
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                })
            except Exception as fallback_e:
                logger.error(f"Chat - Transformers fallback failed: {fallback_e}", exc_info=True)
                return jsonify({"error": {"message": f"Chat unavailable: {fallback_e}", "type": "server_error"}}), 500


        # Format chat messages into a single prompt string suitable for the model
        # This might need adjustment based on the specific model's chat template
        # Using a generic approach here - consider AutoTokenizer.apply_chat_template if available/compatible
        tokenizer = None
        try:
            tokenizer = _load_tokenizer(model_path)
        except Exception as tokenizer_e:
            logger.warning(f"Chat - Tokenizer load failed; using fallback prompt. Details: {tokenizer_e}")

        prompt = _build_chat_prompt(messages, tokenizer)

        logger.info(f"Chat Request - Formatted Prompt (first 200 chars): {prompt[:200]}...")


        # Log memory before generation (if GPU)
        if device_type in ('cuda', 'rocm'):
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"Chat - GPU memory before generation - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


        # Create SGLang generation state
        if repetition_penalty != 1.0 or no_repeat_ngram_size or top_k or typical_p != 1.0:
            logger.info(
                "SGLang chat generation: repetition_penalty/no_repeat_ngram_size/top_k/typical_p are not applied "
                "(not supported via RuntimeState here)."
            )
        state = sgl.RuntimeState(
            temperature=temperature, # Use requested temp
            top_p=top_p,
            max_new_tokens=max_tokens # Use requested max_tokens
        )
        logger.info(f"Chat - SGLang RuntimeState created: temp={state.temperature}, top_p={state.top_p}, max_new_tokens={state.max_new_tokens}")

        # Generate response using SGLang
        try:
            response_text = sgl_model.generate(prompt=prompt, state=state)
            logger.info("Chat - SGLang generation successful.")
        except Exception as sgl_error:
            logger.error(f"Chat - SGLang generation error: {sgl_error}", exc_info=True)
            return jsonify({"error": {"message": f"SGLang chat generation failed: {sgl_error}", "type": "server_error"}}), 500

        # Log memory after generation (if GPU)
        if device_type in ('cuda', 'rocm'):
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"Chat - GPU memory after generation - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        # Log the raw model response
        logger.info(f"Chat - Raw Model response (first 100 chars): {response_text[:100]}...")

        # Format response in OpenAI style
        completion_timestamp = int(datetime.now().timestamp())
        # Simple token count estimation (replace with precise method if needed)
        if tokenizer:
            prompt_tokens = len(tokenizer(prompt).input_ids)
            completion_tokens = len(tokenizer(response_text).input_ids)
        else:
            prompt_tokens = len(prompt.split())
            completion_tokens = len(response_text.split())

        return jsonify({
            "id": f"chatcmpl-{completion_timestamp}",
            "object": "chat.completion",
            "created": completion_timestamp,
            "model": model_id_openai, # Return the requested model name
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text.strip() # Use the generated text
                    },
                    "finish_reason": "stop" # SGLang doesn't directly provide this, assume stop
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        })

    except Exception as e:
        logger.error(f"Chat Completions endpoint error: {str(e)}", exc_info=True) # Log traceback
        return jsonify({"error": {"message": str(e), "type": "server_error", "code": "internal_error"}}), 500

@bp.route('/add', methods=['POST'])
def add_model():
    try:
        data = request.json
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({
                "status": "error",
                "message": "Model ID is required"
            }), 400

        # Fetch model info from Huggingface API
        headers = {}
        if model_manager.hf_token:
            headers["Authorization"] = f"Bearer {model_manager.hf_token}"
            
        response = requests.get(
            f"https://huggingface.co/api/models/{model_id}",
            headers=headers
        )
        response.raise_for_status()
        model_data = response.json()

        # Get parameter count
        parameters = model_data.get('safetensors', {}).get('total')
        if not parameters:
            # Fallback to config-based estimation
            config = model_data.get('config', {})
            if config.get('architectures') == ['GPT2LMHeadModel']:
                n_layer = config.get('n_layer', 12)
                n_embd = config.get('n_embd', 768)
                parameters = 12 * n_layer * (12 * n_embd**2 + 13 * n_embd)
            elif 'llama' in model_id.lower():
                hidden_size = config.get('hidden_size', 4096)
                num_hidden_layers = config.get('num_hidden_layers', 32)
                parameters = (hidden_size * num_hidden_layers * 6 * 1024)

        # Format parameters
        if parameters:
            if parameters >= 1e9:
                param_str = f"{parameters/1e9:.1f}B"
                size_category = "medium" if parameters < 10e9 else "large"
            elif parameters >= 1e6:
                param_str = f"{parameters/1e6:.1f}M"
                size_category = "small"
            else:
                param_str = f"{parameters/1e3:.1f}K"
                size_category = "small"
        else:
            param_str = "Unknown"
            size_category = "medium"  # Default to medium if unknown

        # Get storage size
        storage_bytes = model_data.get('usedStorage')
        if storage_bytes:
            if storage_bytes >= 1e9:
                storage_size = f"{storage_bytes / 1e9:.1f}GB"
            elif storage_bytes >= 1e6:
                storage_size = f"{storage_bytes / 1e6:.1f}MB"
            else:
                storage_size = f"{storage_bytes / 1e3:.1f}KB"
        else:
            # Fallback to calculating from siblings
            siblings = model_data.get('siblings', [])
            total_size = sum(s.get('size', 0) for s in siblings)
            if total_size >= 1e9:
                storage_size = f"{total_size / 1e9:.1f}GB"
            elif total_size >= 1e6:
                storage_size = f"{total_size / 1e6:.1f}MB"
            else:
                storage_size = f"{total_size / 1e3:.1f}KB"

        # Get display name
        display_name = data.get('display_name', '')
        model_name = display_name or model_data['id'].split('/')[-1].replace('-', ' ').title()

        # Build model configuration
        new_model = {
            "id": model_id,
            "name": model_name,
            "size_category": size_category,
            "parameters": param_str or "Unknown",
            "storage_size": storage_size or "Unknown",
            "description": (model_data.get('description', '')[:100] if model_data else '') or "No description available",
            "supports_lora": True,
            "requirements": {
                "min_gpu_memory": {
                    "standard": "16GB",
                    "lora": "8GB",
                    "qlora": "4GB"
                },
                "recommended_batch_size": {
                    "cuda": {
                        "standard": 1,
                        "lora": 2,
                        "qlora": 4
                    },
                    "rocm": {
                        "standard": 1,
                        "lora": 2,
                        "qlora": 4
                    },
                    "cpu": 1
                }
            },
            "custom": True  # Mark as custom model
        }

        # Update config.json
        config_path = Path(__file__).parent.parent / 'config.json'
        with open(config_path, 'r+') as f:
            config = json.load(f)
            if any(m['id'] == model_id for m in config['models']):
                return jsonify({
                    "status": "error",
                    "message": "Model already exists"
                }), 400
                
            config['models'].append(new_model)
            f.seek(0)
            json.dump(config, f, indent=2)
            f.truncate()

        return jsonify({
            "status": "success",
            "message": "Model added successfully",
            "model": new_model
        })
        
    except Exception as e:
        logger.error(f"Error adding model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@bp.route('/downloaded/<path:model_id>', methods=['DELETE'])
def delete_downloaded_model(model_id):
    try:
        safe_dir_name = model_id.replace('/', '_')
        model_path = (MODELS_DIR / safe_dir_name).resolve()
        base_dir = MODELS_DIR.resolve()
        if base_dir not in model_path.parents and model_path != base_dir:
            return jsonify({
                "status": "error",
                "message": "Invalid model path"
            }), 400

        if not model_path.exists():
            return jsonify({
                "status": "error",
                "message": "Downloaded model not found"
            }), 404

        if not model_path.is_dir():
            return jsonify({
                "status": "error",
                "message": "Downloaded model path is not a directory"
            }), 400

        shutil.rmtree(model_path)
        sgl_runtime_cache.pop(str(model_path), None)
        logger.info(f"Deleted downloaded model files at {model_path}")
        return jsonify({
            "status": "success",
            "message": "Downloaded model files deleted"
        })
    except Exception as e:
        logger.error(f"Error deleting downloaded model files: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@bp.route('/delete/<path:model_id>', methods=['DELETE'])
def delete_model(model_id):
    try:
        config_path = Path(__file__).parent.parent / 'config.json'
        with open(config_path, 'r+') as f:
            config = json.load(f)
            new_models = [m for m in config['models'] if m['id'] != model_id]
            
            if len(new_models) == len(config['models']):
                return jsonify({
                    "status": "error",
                    "message": "Model not found"
                }), 404
                
            config['models'] = new_models
            f.seek(0)
            json.dump(config, f, indent=2)
            f.truncate()

        return jsonify({
            "status": "success",
            "message": "Model deleted successfully"
        })
        
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500 

    