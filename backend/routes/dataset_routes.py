from flask import Blueprint, jsonify, request
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from config.settings import DATASETS_DIR, CONFIG_DIR
import logging

bp = Blueprint('datasets', __name__, url_prefix='/api/datasets')
logger = logging.getLogger('training')

def analyze_file_structure(file_path):
    ext = Path(file_path).suffix.lower()
    
    try:
        if ext in ['.json', '.jsonl']:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                if ext == '.jsonl':
                    # For JSONL, try multiple lines if first line fails
                    for _ in range(5):  # Try first 5 lines
                        line = f.readline().strip()
                        if not line:
                            continue
                        try:
                            sample = json.loads(line)
                            if isinstance(sample, dict):
                                break
                        except json.JSONDecodeError:
                            continue
                    if not sample:
                        raise ValueError("Could not find valid JSON line in JSONL file")
                else:
                    # For JSON, try multiple strategies
                    sample = None
                    errors = []

                    # Strategy 1: Try to find a complete object in first few lines
                    f.seek(0)
                    buffer = ""
                    for _ in range(10):  # Try first 10 lines
                        buffer += f.readline()
                        try:
                            # Try to find a complete object
                            buffer = buffer.strip()
                            if buffer.startswith('{'):
                                end_idx = buffer.find('}')
                                if end_idx > 0:
                                    potential_obj = buffer[:end_idx + 1]
                                    sample = json.loads(potential_obj)
                                    break
                            elif buffer.startswith('[{'):
                                end_idx = buffer.find('}]')
                                if end_idx > 0:
                                    potential_obj = buffer[1:end_idx + 1]  # Remove outer brackets
                                    sample = json.loads(potential_obj)
                                    break
                        except json.JSONDecodeError as e:
                            errors.append(f"Line parsing error: {str(e)}")
                            continue

                    # Strategy 2: If still no sample, try reading chunks
                    if not sample:
                        f.seek(0)
                        chunk_size = 4096  # Smaller chunks
                        content = ''
                        for _ in range(5):  # Try up to 5 chunks
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            content += chunk
                            try:
                                # Try to find complete objects
                                if '{' in content and '}' in content:
                                    start = content.find('{')
                                    end = content.find('}', start) + 1
                                    if start >= 0 and end > start:
                                        potential_obj = content[start:end]
                                        sample = json.loads(potential_obj)
                                        break
                            except json.JSONDecodeError as e:
                                errors.append(f"Chunk parsing error: {str(e)}")
                                continue

                    if not sample:
                        error_msg = "\n".join(errors)
                        raise ValueError(f"Could not parse JSON content. Errors:\n{error_msg}")

                # Validate and return structure
                if not isinstance(sample, dict):
                    sample = {"text": str(sample)}  # Fallback for non-object values

                return {
                    'type': 'json',
                    'fields': list(sample.keys()),
                    'sample': sample
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
        dataset_path = DATASETS_DIR / file.filename
        file.save(dataset_path)
        
        file_structure = analyze_file_structure(dataset_path)
        stats = os.stat(dataset_path)
        
        return jsonify({
            "status": "success",
            "message": "Dataset prepared successfully",
            "dataset_path": str(dataset_path),
            "file_info": {
                "name": file.filename,
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
                    dataset_name = config_file.stem.replace('.config', '')
                    dataset_path = DATASETS_DIR / f"{dataset_name}.json"
                    
                    if dataset_path.exists():
                        stats = os.stat(dataset_path)
                        configured_datasets.append({
                            "name": dataset_name,
                            "path": str(dataset_path),
                            "size": stats.st_size,
                            "uploadedAt": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                            "config": dataset_config
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
                    datasets.append({
                        "name": file_path.name,
                        "path": str(file_path),
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
    file_path = data.get('file_path')
    
    if not file_path:
        return jsonify({
            "status": "error",
            "message": "File path is required"
        }), 400
        
    try:
        structure = analyze_file_structure(Path(file_path))
        return jsonify({
            "status": "success",
            "structure": structure
        })
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

        file_path = data.get('file_path')
        config = data.get('config')
        
        logger.info(f"Received config save request for {file_path}: {config}")
        
        if not file_path or not config:
            logger.error("Missing file_path or config in request")
            return jsonify({
                "status": "error",
                "message": "File path and config are required"
            }), 400
            
        if not config.get('inputField') or not config.get('outputField'):
            logger.error("Missing input or output field in config")
            return jsonify({
                "status": "error",
                "message": "Input and output fields are required"
            }), 400

        # Create config directory if it doesn't exist
        CONFIG_DIR.mkdir(exist_ok=True)
        
        # Save config file using the dataset filename without the extension
        dataset_name = Path(file_path).stem
        config_path = CONFIG_DIR / f"{dataset_name}.config.json"
        
        logger.info(f"Saving config to {config_path}")
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info("Config saved successfully")
        
        return jsonify({
            "status": "success",
            "message": "Configuration saved successfully",
            "config_path": str(config_path),
            "saved_config": config
        })
        
    except Exception as e:
        logger.error(f"Error saving dataset config: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Add other dataset routes... 