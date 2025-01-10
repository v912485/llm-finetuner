import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import MODELS_DIR
import logging
from flask import jsonify

logger = logging.getLogger('training')

class ModelManager:
    def __init__(self):
        self.config_path = Path(__file__).parent.parent / 'config.json'
        self.hf_token = os.environ.get('HUGGING_FACE_TOKEN')
        
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {"models": []}
            
    def get_available_models(self):
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
        try:
            safe_dir_name = model_id.replace('/', '_')
            model_path = MODELS_DIR / safe_dir_name
            
            # Add token for gated models
            auth_token = self.hf_token if 'google/gemma' in model_id else None
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                token=auth_token
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=auth_token
            )
            
            model_path.mkdir(exist_ok=True, parents=True)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            return jsonify({
                "status": "success",
                "message": f"Model {model_id} downloaded successfully"
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Failed to download model: {str(e)}"
            }), 500 