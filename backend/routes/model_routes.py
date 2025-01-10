from flask import Blueprint, jsonify, request
from models.model_manager import ModelManager
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import MODELS_DIR, SAVED_MODELS_DIR
import logging
import torch
import json

bp = Blueprint('models', __name__, url_prefix='/api/models')
logger = logging.getLogger('training')
model_manager = ModelManager()

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
        data = request.json
        logger.info(f"Received inference request with data: {data}")
        
        model_id = data.get('model_id')
        input_text = data.get('input')
        use_finetuned = data.get('use_finetuned', False)
        saved_model_name = data.get('saved_model_name')
        
        if not model_id or not input_text:
            logger.error(f"Missing required fields. model_id: {model_id}, input: {input_text}")
            return jsonify({
                "status": "error",
                "message": "Model ID and input text are required"
            }), 400
        
        # Get model path
        if saved_model_name:
            model_path = SAVED_MODELS_DIR / saved_model_name
            logger.info(f"Using saved model: {saved_model_name}")
        else:
            safe_model_name = model_id.replace('/', '_')
            model_path = MODELS_DIR / safe_model_name
            if use_finetuned:
                model_path = model_path / 'finetuned'
                logger.info(f"Using fine-tuned model")
            
        if not model_path.exists():
            return jsonify({
                "status": "error",
                "message": f"Model not found at {model_path}"
            }), 404
            
        # Get device info
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_info = {
            'type': 'cuda' if torch.cuda.is_available() else 'cpu',
            'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'memory': f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB" if torch.cuda.is_available() else 'N/A'
        }
        logger.info(f"Using device: {device_info['name']} ({device_info['type'].upper()})")
        if device_info['type'] == 'cuda':
            logger.info(f"GPU Memory: {device_info['memory']}")
        
        # Load model and tokenizer with explicit device mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,  # Set device explicitly
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,  # Use fp16 for GPU
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        logger.info(f"Model loaded successfully on {model.device}")
        
        # Generate response
        inputs = tokenizer(input_text, return_tensors="pt").to(device)  # Move inputs to device
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            "status": "success",
            "response": response,
            "device_info": device_info
        })
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500 

@bp.route('/saved', methods=['GET'])
def get_saved_models():
    try:
        saved_models = []
        
        if SAVED_MODELS_DIR.exists():
            for model_dir in SAVED_MODELS_DIR.iterdir():
                if model_dir.is_dir():
                    metadata_path = model_dir / 'metadata.json'
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            saved_models.append({
                                'name': model_dir.name,
                                'original_model': metadata['original_model'],
                                'save_date': metadata['save_date'],
                                'path': str(model_dir)
                            })
        
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