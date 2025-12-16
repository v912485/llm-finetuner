from flask import Blueprint, jsonify, request
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from config.settings import DATASETS_DIR, CONFIG_DIR
import logging
from uuid import uuid4
from werkzeug.utils import secure_filename
from utils.datasets import resolve_dataset_path, split_dataset_filename, load_json_dataset

bp = Blueprint('datasets', __name__, url_prefix='/api/datasets')
logger = logging.getLogger('training')

def analyze_file_structure(file_path):
    ext = Path(file_path).suffix.lower()
    
    try:
        if ext in ['.json', '.jsonl']:
            dataset_path = Path(file_path)
            detected_type = 'jsonl' if ext == '.jsonl' else 'json'

            if ext == '.json':
                try:
                    with open(dataset_path, 'r', encoding='utf-8', errors='replace') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    detected_type = 'jsonl'

            records = load_json_dataset(dataset_path)
            sample = None
            for entry in records[:50]:
                if isinstance(entry, dict):
                    sample = entry
                    break

            if sample is None:
                sample = records[0] if records else {}

            if not isinstance(sample, dict):
                sample = {"text": str(sample)}

            is_messages_format = False
            if 'messages' in sample and isinstance(sample['messages'], list):
                if sample['messages'] and isinstance(sample['messages'][0], dict):
                    if 'role' in sample['messages'][0] and 'content' in sample['messages'][0]:
                        is_messages_format = True

            return {
                'type': detected_type,
                'fields': list(sample.keys()),
                'sample': sample,
                'is_messages_format': is_messages_format
            }
                
        elif ext == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to detect dialect
                sample = f.read(1024)
                dialect = csv.Sniffer().sniff(sample)
                f.seek(0)
                
                reader = csv.reader(f, dialect)
                headers = next(reader)
                
                # Read a few rows to suggest field types
                rows = []
                for _ in range(5):
                    try:
                        rows.append(next(reader))
                    except StopIteration:
                        break
                
                field_info = []
                for i, header in enumerate(headers):
                    values = [row[i] for row in rows if len(row) > i]
                    field_type = 'text'  # default type
                    
                    # Try to detect field type
                    if values:
                        if all(v.replace('.','',1).isdigit() for v in values):
                            field_type = 'number'
                        elif all(v.lower() in ['true', 'false', '1', '0'] for v in values):
                            field_type = 'boolean'
                            
                    field_info.append({
                        'name': header,
                        'type': field_type,
                        'sample': values[:3] if values else []
                    })
                
                return {
                    'type': 'csv',
                    'fields': headers,
                    'field_info': field_info,
                    'dialect': {
                        'delimiter': dialect.delimiter,
                        'quotechar': dialect.quotechar
                    }
                }
                
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first few lines to detect structure
                lines = []
                for _ in range(5):
                    line = f.readline().strip()
                    if line:
                        lines.append(line)
                    if not line:
                        break
                
                # Try to detect if it's a conversation format
                is_conversation = any(line.startswith(('User:', 'Assistant:', 'Human:', 'AI:', '>', '<')) for line in lines)
                
                return {
                    'type': 'text',
                    'format': 'conversation' if is_conversation else 'plain',
                    'fields': ['input', 'output'] if is_conversation else ['text'],
                    'sample_lines': lines
                }
                
        return {'type': 'unknown', 'fields': []}
        
    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}")
        return {'type': 'error', 'fields': [], 'error': str(e)}

@bp.route('/prepare', methods=['POST'])
def prepare_dataset():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400
    
    try:
        dataset_id = uuid4().hex
        original_name = secure_filename(file.filename) or "dataset"
        stored_name = f"{dataset_id}_{original_name}"
        dataset_path = DATASETS_DIR / stored_name
        file.save(dataset_path)
        
        file_structure = analyze_file_structure(dataset_path)
        stats = os.stat(dataset_path)
        
        return jsonify({
            "status": "success",
            "message": "Dataset prepared successfully",
            "dataset_id": dataset_id,
            "file_info": {
                "name": original_name,
                "size": stats.st_size,
                "uploadedAt": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "structure": file_structure
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route('/configured', methods=['GET'])
def get_configured_datasets():
    try:
        configured_datasets = []
        if CONFIG_DIR.exists():
            for config_file in CONFIG_DIR.glob('*.config.json'):
                with open(config_file, 'r') as f:
                    dataset_config = json.load(f)
                    dataset_id = config_file.stem.replace('.config', '')
                    try:
                        dataset_path = resolve_dataset_path(DATASETS_DIR, dataset_id)
                    except Exception:
                        continue
                    
                    if dataset_path.exists():
                        stats = os.stat(dataset_path)
                        parsed_id, display_name = split_dataset_filename(dataset_path.name)
                        safe_config = {
                            "input_field": dataset_config.get("input_field"),
                            "output_field": dataset_config.get("output_field"),
                            "created_at": dataset_config.get("created_at"),
                        }
                        configured_datasets.append({
                            "dataset_id": dataset_id,
                            "name": display_name,
                            "size": stats.st_size,
                            "uploadedAt": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                            "config": safe_config
                        })
        
        logger.info(f"Found configured datasets: {configured_datasets}")
        return jsonify({
            "status": "success",
            "configured_datasets": configured_datasets
        })
        
    except Exception as e:
        logger.error(f"Error getting configured datasets: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@bp.route('/downloaded', methods=['GET'])
def get_downloaded_datasets():
    try:
        datasets = []
        if DATASETS_DIR.exists():
            for file_path in DATASETS_DIR.iterdir():
                if file_path.is_file():
                    stats = os.stat(file_path)
                    dataset_id, display_name = split_dataset_filename(file_path.name)
                    datasets.append({
                        "dataset_id": dataset_id,
                        "name": display_name,
                        "size": stats.st_size,
                        "uploadedAt": datetime.fromtimestamp(stats.st_mtime).isoformat()
                    })
        
        return jsonify({
            "status": "success",
            "datasets": datasets
        })
        
    except Exception as e:
        logger.error(f"Error getting downloaded datasets: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@bp.route('/structure', methods=['POST'])
def get_file_structure():
    data = request.json
    dataset_id = data.get('dataset_id')
    
    if not dataset_id:
        return jsonify({
            "status": "error",
            "message": "dataset_id is required"
        }), 400
        
    try:
        dataset_path = resolve_dataset_path(DATASETS_DIR, dataset_id)
        structure = analyze_file_structure(dataset_path)
        return jsonify({
            "status": "success",
            "structure": structure
        })
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        logger.error(f"Error analyzing file structure: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@bp.route('/config', methods=['POST'])
def save_dataset_config():
    try:
        data = request.json
        if not data:
            logger.error("No JSON data received")
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400

        dataset_id = data.get('dataset_id')
        config = data.get('config')
        
        logger.info(f"Received config save request for dataset_id={dataset_id}: {config}")
        
        if not dataset_id or not config:
            logger.error("Missing dataset_id or config in request")
            return jsonify({
                "status": "error",
                "message": "dataset_id and config are required"
            }), 400
            
        if not config.get('inputField') or not config.get('outputField'):
            logger.error("Missing input or output field in config")
            return jsonify({
                "status": "error",
                "message": "Input and output fields are required"
            }), 400

        # Create config directory if it doesn't exist
        CONFIG_DIR.mkdir(exist_ok=True)
        
        config_path = CONFIG_DIR / f"{dataset_id}.config.json"
        
        logger.info(f"Saving config to {config_path}")
        
        # Save full configuration
        full_config = {
            'dataset_id': dataset_id,
            'input_field': config['inputField'],
            'output_field': config['outputField'],
            'created_at': datetime.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)
            
        logger.info("Config saved successfully")
        
        return jsonify({
            "status": "success",
            "message": "Configuration saved successfully",
            "dataset_id": dataset_id,
            "saved_config": full_config
        })
        
    except Exception as e:
        logger.error(f"Error saving dataset config: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Add other dataset routes... 