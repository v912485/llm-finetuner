from flask import Blueprint, jsonify, request
from training.trainer import Trainer
import logging
import json
from datetime import datetime
from pathlib import Path
from config.settings import MODELS_DIR, SAVED_MODELS_DIR
import shutil

bp = Blueprint('training', __name__, url_prefix='/api/training')
logger = logging.getLogger('training')
trainer = Trainer()

@bp.route('/start', methods=['POST'])
def start_training():
    logger.info("Received training start request")
    
    if trainer.is_training():
        logger.warning("Training already in progress")
        return jsonify({
            "status": "error",
            "message": "Training is already in progress"
        }), 400
    
    data = request.json
    logger.info(f"Training request data: {data}")
    
    if not data.get('model_id') or not data.get('datasets'):
        logger.error("Missing required training parameters")
        return jsonify({
            "status": "error",
            "message": "Model ID and datasets are required"
        }), 400
    
    try:
        trainer.start_training(data)
        logger.info("Training started successfully")
        return jsonify({
            "status": "success",
            "message": "Training started successfully"
        })
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@bp.route('/status', methods=['GET'])
def get_training_status():
    try:
        status = trainer.get_status()
        return jsonify({
            "status": "success",
            **status
        })
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500 

@bp.route('/save', methods=['POST'])
def save_model():
    try:
        data = request.json
        model_id = data.get('model_id')
        save_name = data.get('save_name')
        
        if not model_id or not save_name:
            return jsonify({
                "status": "error",
                "message": "Model ID and save name are required"
            }), 400
            
        # Sanitize save name
        safe_save_name = "".join(c for c in save_name if c.isalnum() or c in ('-', '_')).strip()
        if not safe_save_name:
            return jsonify({
                "status": "error",
                "message": "Invalid save name"
            }), 400
            
        # Get paths
        safe_model_name = model_id.replace('/', '_')
        source_path = MODELS_DIR / safe_model_name / 'finetuned'
        target_path = SAVED_MODELS_DIR / safe_save_name
        
        if not source_path.exists():
            return jsonify({
                "status": "error",
                "message": "No fine-tuned model found"
            }), 404
            
        # Create saved models directory
        SAVED_MODELS_DIR.mkdir(exist_ok=True)
        
        if target_path.exists():
            return jsonify({
                "status": "error",
                "message": "A model with this name already exists"
            }), 409
            
        # Copy model files
        shutil.copytree(source_path, target_path)
        
        # Save metadata
        metadata = {
            'original_model': model_id,
            'save_date': datetime.now().isoformat(),
            'training_params': trainer.get_status()
        }
        
        with open(target_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Model saved successfully as {safe_save_name}")
        
        return jsonify({
            "status": "success",
            "message": "Model saved successfully",
            "save_path": str(target_path)
        })
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500 