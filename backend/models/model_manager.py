import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import MODELS_DIR, SAVED_MODELS_DIR, HF_TOKEN
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
import logging
from flask import jsonify
from training.trainer_instance import trainer
import torch

logger = logging.getLogger('training')

class ModelManager:
    def __init__(self):
        self.config_path = Path(__file__).parent.parent / 'config.json'
        self.hf_token = HF_TOKEN  # Use token from settings
        self.trainer = trainer
        self.available_models = self.get_available_models()
        
        if not self.hf_token:
            logger.warning("No Hugging Face token found. Some models may not be accessible.")
        
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {"models": []}
            
    def get_available_models(self):
        """Get list of available models from config"""
        return self.load_config().get('models', [])
        
    def get_downloaded_models(self):
        try:
            downloaded = []
            if not MODELS_DIR.exists():
                return {"status": "success", "downloaded_models": []}

            for model_dir in os.listdir(MODELS_DIR):
                model_path = MODELS_DIR / model_dir
                if model_path.is_dir():
                    has_config = (model_path / 'config.json').exists()
                    if not has_config:
                        continue

                    has_tokenizer = any(
                        (model_path / f).exists()
                        for f in (
                            'tokenizer.json',
                            'tokenizer.model',
                            'vocab.json',
                            'merges.txt',
                            'tokenizer_config.json',
                        )
                    )

                    has_weights = (
                        (model_path / 'pytorch_model.bin').exists()
                        or (model_path / 'pytorch_model.bin.index.json').exists()
                        or (model_path / 'model.safetensors').exists()
                        or (model_path / 'model.safetensors.index.json').exists()
                        or any(model_path.glob('*.safetensors'))
                    )

                    if has_weights and has_tokenizer:
                        model_id = model_dir.replace('_', '/')
                        downloaded.append(model_id)
                        
            return {"status": "success", "downloaded_models": downloaded}
        except Exception as e:
            logger.error(f"Error in get_downloaded_models: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def download_model(self, model_id):
        """Download a model from Hugging Face"""
        try:
            if not self.hf_token:
                return jsonify({
                    "status": "error",
                    "message": "Hugging Face token not found in environment variables"
                }), 401

            logger.info(f"Starting download of model: {model_id}")
            
            # Convert model ID to safe directory name
            safe_dir_name = model_id.replace('/', '_')
            model_path = MODELS_DIR / safe_dir_name
            temp_path = MODELS_DIR / f"{safe_dir_name}_temp"
            
            # Create a temporary directory for download
            temp_path.mkdir(parents=True, exist_ok=True)
            
            # Download model and tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    token=self.hf_token,
                    trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=self.hf_token,
                    trust_remote_code=True,
                    dtype=torch.float32  # Start with float32 for compatibility
                )
                
                # Save model and tokenizer with same parameters used for loading
                model.save_pretrained(temp_path, safe_serialization=True)
                tokenizer.save_pretrained(temp_path)
                
                # If successful, remove existing directory and rename temp
                if model_path.exists():
                    import shutil
                    shutil.rmtree(model_path)
                temp_path.rename(model_path)
                
                logger.info(f"Successfully downloaded model to {model_path}")
                
                return jsonify({
                    "status": "success",
                    "message": "Model downloaded successfully"
                })
                
            except Exception as e:
                # Clean up temp directory on failure
                import shutil
                if temp_path.exists():
                    shutil.rmtree(temp_path)
                logger.error(f"Error downloading model: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": f"Error downloading model: {str(e)}"
                }), 500
                
        except Exception as e:
            logger.error(f"Error in download_model: {str(e)}")
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
            
    def cancel_training(self):
        """Cancel ongoing training"""
        if not self.trainer.is_training():
            raise ValueError("No training in progress")
        self.trainer.cancel_training()

    def generate_response(self, model_id, input_text, temperature=0.7, max_length=512, top_p=0.95):
        """Generate a response using the specified model with OpenAI-compatible parameters"""
        try:
            # Check if it's a saved model name first
            model_path = SAVED_MODELS_DIR / model_id
            if not model_path.exists():
                # Fallback to base model
                safe_dir_name = model_id.replace('/', '_')
                model_path = MODELS_DIR / safe_dir_name
            
            if not model_path.exists():
                raise ValueError(f"Model {model_id} not found. Please download it first.")
            
            # Load model and tokenizer
            is_peft = (model_path / "adapter_config.json").exists()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if is_peft:
                logger.info(f"Loading LoRA adapter from {model_path}")
                # Load metadata to find the base model
                base_model_id = None
                metadata_path = model_path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        base_model_id = metadata.get('original_model') or metadata.get('base_model_id')
                
                if not base_model_id:
                    raise ValueError(f"Could not determine base model for adapter at {model_path}")

                safe_base_name = base_model_id.replace('/', '_')
                base_model_path = MODELS_DIR / safe_base_name
                
                if not base_model_path.exists():
                    raise ValueError(f"Base model {base_model_id} not found at {base_model_path}")

                tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    trust_remote_code=True,
                    dtype=torch.float32
                )
                if PeftModel is None:
                    raise ValueError("PEFT library not installed; cannot load LoRA adapter.")
                model = PeftModel.from_pretrained(base_model, model_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=None,
                    trust_remote_code=True,
                    dtype=torch.float32
                )
            
            # Move model to device
            model = model.to(device)
            
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the response
            if response.startswith(input_text):
                response = response[len(input_text):]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise 