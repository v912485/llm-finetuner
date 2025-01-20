from flask import Blueprint, jsonify, request
from models.model_manager import ModelManager
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import MODELS_DIR, SAVED_MODELS_DIR
import logging
import torch
import json
from datetime import datetime

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
        
        # Get generation parameters
        temperature = float(data.get('temperature', 0.7))
        max_length = int(data.get('max_length', 512))
        do_sample = bool(data.get('do_sample', True))
        
        logger.info(f"Generation parameters: temp={temperature}, max_length={max_length}, do_sample={do_sample}")
        
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
            
        if not model_id and not saved_model_name:
            logger.error("Neither model_id nor saved_model_name provided")
            return jsonify({
                "status": "error",
                "message": "Either model_id or saved_model_name is required"
            }), 400
            
        # Get model path and load model/tokenizer
        try:
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
                
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side='left'  # Add padding to the left
            )
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Get device info
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device_info = {
                'type': 'cuda' if torch.cuda.is_available() else 'cpu',
                'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'memory': f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB" if torch.cuda.is_available() else 'N/A'
            }
            
            # Load model with proper configuration
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                trust_remote_code=True
            )
            
            # Ensure model is in eval mode
            model.eval()
            
            # Tokenize input with proper padding
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length // 2  # Leave room for generation
            ).to(device)
            
            # Generate with safety checks
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(max_length, model.config.max_position_embeddings - inputs['input_ids'].shape[1]),
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # Decode response
            response = tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return jsonify({
                "status": "success",
                "response": response,
                "device_info": device_info
            })
            
        except Exception as model_error:
            logger.error(f"Model error: {str(model_error)}")
            return jsonify({
                "status": "error",
                "message": f"Error processing request: {str(model_error)}"
            }), 500
            
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
    """OpenAI-compatible chat completions endpoint"""
    try:
        data = request.json
        
        # Extract OpenAI-style parameters
        messages = data.get('messages', [])
        if not messages:
            return jsonify({
                "error": {
                    "message": "messages is required",
                    "type": "invalid_request_error",
                    "code": "invalid_messages"
                }
            }), 400
            
        # Get model parameters
        model = data.get('model')  # This will be mapped to our model ID
        temperature = float(data.get('temperature', 0.7))
        max_tokens = int(data.get('max_tokens', 512))
        top_p = float(data.get('top_p', 0.95))
        
        # Convert chat format to text
        prompt = ""
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'system':
                prompt += f"System: {content}\n"
            elif role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        
        # Check if we're using a saved model
        model_path = None
        if model:
            if SAVED_MODELS_DIR.exists():
                saved_model_path = SAVED_MODELS_DIR / model
                if saved_model_path.exists():
                    model_path = saved_model_path
                    logger.info(f"Using saved model: {model}")
        
        if not model_path:
            # If not a saved model, treat as a regular model ID
            safe_model_name = model.replace('/', '_')
            model_path = MODELS_DIR / safe_model_name
            logger.info(f"Using base model: {model}")
        
        if not model_path.exists():
            return jsonify({
                "error": {
                    "message": f"Model not found at {model_path}",
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }), 404
            
        # Load tokenizer and model
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side='left'
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_instance = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                trust_remote_code=True
            )
            
            model_instance.eval()
            
            # Tokenize and generate
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens // 2
            ).to(device)
            
            with torch.no_grad():
                outputs = model_instance.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, model_instance.config.max_position_embeddings - inputs['input_ids'].shape[1]),
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Remove the prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt):]
            
            # Format response in OpenAI style
            completion_timestamp = int(datetime.now().timestamp())
            return jsonify({
                "id": f"chatcmpl-{completion_timestamp}",
                "object": "chat.completion",
                "created": completion_timestamp,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.strip()
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response.split()),
                    "total_tokens": len(prompt.split()) + len(response.split())
                }
            })
            
        except Exception as model_error:
            logger.error(f"Model error: {str(model_error)}")
            return jsonify({
                "error": {
                    "message": f"Error processing request: {str(model_error)}",
                    "type": "model_error",
                    "code": "model_error"
                }
            }), 500
            
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "internal_error"
            }
        }), 500 