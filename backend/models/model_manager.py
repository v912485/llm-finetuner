import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import MODELS_DIR, HF_TOKEN
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
                    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
                    if any((model_path / file).exists() for file in required_files):
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
            
            # Create models directory if it doesn't exist
            MODELS_DIR.mkdir(exist_ok=True)
            
            # Convert model ID to safe directory name
            safe_dir_name = model_id.replace('/', '_')
            model_path = MODELS_DIR / safe_dir_name
            
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
                    torch_dtype=torch.float32  # Start with float32 for compatibility
                )
                
                # Save model and tokenizer
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                
                logger.info(f"Successfully downloaded model to {model_path}")
                
                return jsonify({
                    "status": "success",
                    "message": "Model downloaded successfully"
                })
                
            except Exception as e:
                logger.error(f"Error downloading model: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": f"Failed to download model: {str(e)}"
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