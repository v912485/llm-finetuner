from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm
import sys
import csv
from pathlib import Path
from threading import Thread
import time
import logging
from datetime import datetime
import os.path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import torch.cuda
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

MODELS_DIR = "downloaded_models"
DATASETS_DIR = "datasets"
CONFIG_DIR = "dataset_configs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Set up logging
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Set up file handler
    log_file = os.path.join('logs', f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Set up logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Add global variable to store training status
training_status = {
    'is_training': False,
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'loss': None,
    'error': None,
    'start_time': None,
    'end_time': None,
    'model_id': None,
    'dataset_info': None,
    'current_step': None,
    'total_steps': None,
    'learning_rate': None
}

def log_training_progress(status_update):
    """Log training progress to file and update status"""
    global training_status
    training_status.update(status_update)
    
    # Create log message
    message_parts = []
    if 'current_epoch' in status_update and 'total_epochs' in status_update:
        message_parts.append(f"Epoch: {status_update['current_epoch']}/{status_update['total_epochs']}")
    if 'progress' in status_update:
        message_parts.append(f"Progress: {status_update['progress']}%")
    if 'loss' in status_update and status_update['loss'] is not None:
        message_parts.append(f"Loss: {status_update['loss']:.4f}")
    if 'current_step' in status_update and 'total_steps' in status_update:
        message_parts.append(f"Step: {status_update['current_step']}/{status_update['total_steps']}")
    
    log_message = " - ".join(message_parts)
    
    if 'error' in status_update and status_update['error']:
        logger.error(f"Training error: {status_update['error']}")
    else:
        logger.info(log_message)

def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {"models": []}  # Return empty config as fallback

# Load config at startup
CONFIG = load_config()

@app.route('/api/models/available', methods=['GET'])
def get_available_models():
    return jsonify(CONFIG.get('models', []))

@app.route('/api/models/download', methods=['POST'])
def download_model():
    data = request.json
    model_id = data.get('model_id')
    
    if not model_id:
        return jsonify({
            "status": "error",
            "message": "Model ID is required"
        }), 400
    
    try:
        # Create a safe directory name
        safe_dir_name = model_id.replace('/', '_')
        model_path = os.path.join(MODELS_DIR, safe_dir_name)
        
        print(f"Downloading model: {model_id} to {model_path}")  # Debug print
        
        # Download model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Save model and tokenizer
        print(f"Saving model to: {model_path}")
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        print(f"Model saved successfully to: {model_path}")  # Debug print
        return jsonify({
            "status": "success",
            "message": f"Model {model_id} downloaded successfully"
        })
    except Exception as e:
        error_message = str(e)
        print(f"Error downloading model: {error_message}")
        return jsonify({
            "status": "error",
            "message": f"Failed to download model: {error_message}"
        }), 500

@app.route('/api/dataset/list', methods=['GET'])
def list_datasets():
    try:
        datasets = []
        for filename in os.listdir(DATASETS_DIR):
            file_path = os.path.join(DATASETS_DIR, filename)
            if os.path.isfile(file_path):
                stats = os.stat(file_path)
                datasets.append({
                    'name': filename,
                    'path': file_path,
                    'size': stats.st_size,
                    'uploadedAt': stats.st_mtime
                })
        return jsonify({
            'status': 'success',
            'datasets': datasets
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def analyze_file_structure(file_path):
    """Analyze the file and return available fields based on file type."""
    ext = Path(file_path).suffix.lower()
    
    try:
        if ext in ['.json', '.jsonl']:
            with open(file_path, 'r') as f:
                # Read first line for JSONL or entire file for JSON
                if ext == '.jsonl':
                    first_line = f.readline()
                    sample = json.loads(first_line)
                else:
                    content = json.load(f)
                    sample = content[0] if isinstance(content, list) else content
                return {
                    'type': 'json',
                    'fields': list(sample.keys())
                }
                
        elif ext == '.csv':
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Get the header row
                return {
                    'type': 'csv',
                    'fields': headers
                }
                
        elif ext == '.txt':
            with open(file_path, 'r') as f:
                content = f.read()
                # Find all markers that look like "### Something:"
                import re
                markers = re.findall(r'###\s+([^:]+):', content)
                return {
                    'type': 'txt',
                    'fields': [f"### {m}:" for m in set(markers)]
                }
                
        return {'type': 'unknown', 'fields': []}
        
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return {'type': 'error', 'fields': []}

@app.route('/api/dataset/prepare', methods=['POST'])
def prepare_dataset():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400
    
    try:
        # Save the dataset file
        dataset_path = os.path.join(DATASETS_DIR, file.filename)
        file.save(dataset_path)
        
        # Analyze file structure
        file_structure = analyze_file_structure(dataset_path)
        
        # Get file stats
        stats = os.stat(dataset_path)
        
        return jsonify({
            "status": "success",
            "message": "Dataset prepared successfully",
            "dataset_path": dataset_path,
            "file_info": {
                "name": file.filename,
                "size": stats.st_size,
                "uploadedAt": stats.st_mtime,
                "structure": file_structure
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    global training_status
    
    if training_status['is_training']:
        return jsonify({
            "status": "error",
            "message": "Training is already in progress"
        }), 400
    
    data = request.json
    if not data.get('model_id') or not data.get('datasets'):
        return jsonify({
            "status": "error",
            "message": "Model ID and datasets are required"
        }), 400
    
    try:
        # Reset training status
        training_status = {
            'is_training': True,
            'progress': 0,
            'current_epoch': 0,
            'total_epochs': data.get('epochs', 3),
            'loss': None,
            'error': None,
            'start_time': None,
            'end_time': None,
            'model_id': data.get('model_id'),
            'dataset_info': data.get('datasets'),
            'current_step': None,
            'total_steps': None,
            'learning_rate': data.get('learningRate', 0.0001)
        }
        
        # Start training in a separate thread
        thread = Thread(target=run_training, args=(data,))
        thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Training started successfully"
        })
    except Exception as e:
        training_status['is_training'] = False
        training_status['error'] = str(e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    return jsonify(training_status)

# Update the CUDA check at the top of app.py
def get_device_info():
    if torch.cuda.is_available():
        return {
            'type': 'cuda',
            'name': torch.cuda.get_device_name(0),
            'backend': 'CUDA'
        }
    elif hasattr(torch, 'hip') and torch.hip.is_available():
        return {
            'type': 'cuda',  # ROCm uses CUDA API
            'name': 'AMD GPU',
            'backend': 'ROCm'
        }
    else:
        return {
            'type': 'cpu',
            'name': 'CPU',
            'backend': 'CPU'
        }

# Initialize device info
DEVICE_INFO = get_device_info()
ACCELERATOR_AVAILABLE = DEVICE_INFO['type'] == 'cuda'

# Update logging at startup
if ACCELERATOR_AVAILABLE:
    logger.info(f"Using {DEVICE_INFO['backend']} on {DEVICE_INFO['name']}")
else:
    logger.warning("No GPU acceleration available. Using CPU for training.")

GRADIENT_ACCUMULATION_STEPS = 8  # Increased from 4
MAX_LENGTH = 128  # Reduced from 256
BATCH_SIZE = 2  # Reduced from 4

def run_training(config):
    global training_status
    try:
        start_time = datetime.now()
        logger.info(f"Starting training with config: {config}")
        
        # Set up device
        device = torch.device(DEVICE_INFO['type'])
        logger.info(f"Training on {DEVICE_INFO['name']} using {DEVICE_INFO['backend']}")
        
        # Load model and tokenizer first
        model_id = config['model_id']
        safe_dir_name = model_id.replace('/', '_')
        model_path = os.path.join(MODELS_DIR, safe_dir_name)
        
        # Set environment variable for memory allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto' if ACCELERATOR_AVAILABLE else None,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Enable gradient checkpointing after model creation
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        elif hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()

        # Disable generation caching for training
        model.config.use_cache = False
        
        # Move model to device if not using auto device mapping
        if not ACCELERATOR_AVAILABLE:
            model = model.to(device)

        # Define CustomDataset class inside the function
        class CustomDataset(Dataset):
            def __init__(self, data, tokenizer, max_length=MAX_LENGTH):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                text = f"{item['input']}{self.tokenizer.sep_token}{item['output']}"
                
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Move tensors to CPU initially
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze()
                }
        
        # Prepare datasets
        train_data, val_data = prepare_training_validation_split(
            config['datasets'],
            config['dataset_configs'],
            validation_split=config.get('validationSplit', 0.2)
        )
        
        logger.info(f"Dataset split: {len(train_data)} training samples, {len(val_data)} validation samples")
        
        # Create datasets with the now-available tokenizer
        train_dataset = CustomDataset(train_data, tokenizer)
        val_dataset = CustomDataset(val_data, tokenizer)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True
        )
        
        # Update training status with device info
        training_status.update({
            'is_training': True,
            'progress': 0,
            'start_time': start_time.isoformat(),
            'model_id': config['model_id'],
            'dataset_info': config['datasets'],
            'learning_rate': config['learningRate'],
            'device': 'GPU' if ACCELERATOR_AVAILABLE else 'CPU'
        })
        
        # Initialize mixed precision training
        scaler = torch.amp.GradScaler('cuda') if ACCELERATOR_AVAILABLE else None
        
        # Setup optimizer with gradient accumulation
        optimizer = AdamW(
            model.parameters(),
            lr=config.get('learningRate', 0.0001),
            eps=1e-8,
            weight_decay=0.01  # Add weight decay for better regularization
        )
        
        num_epochs = config.get('epochs', 3)
        total_steps = (len(train_dataloader) * num_epochs) // GRADIENT_ACCUMULATION_STEPS
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # Training loop with gradient accumulation
        model.train()
        current_step = 0
        optimizer.zero_grad()  # Zero gradients at start
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            
            # Training loop
            for batch_idx, batch in enumerate(train_dataloader):
                if not training_status['is_training']:
                    logger.info("Training cancelled")
                    return

                try:
                    # Move batch to device here
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)

                    # Use gradient scaling with autocast
                    if ACCELERATOR_AVAILABLE:
                        with autocast():
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=input_ids
                            )
                            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                        
                        # Scale loss and backward pass
                        scaler.scale(loss).backward()
                        
                        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            scheduler.step()
                            current_step += 1
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids
                        )
                        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                        loss.backward()
                        
                        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                            scheduler.step()
                            current_step += 1

                    # Clear cache more aggressively
                    if ACCELERATOR_AVAILABLE:
                        torch.cuda.empty_cache()
                        if batch_idx % 5 == 0:  # Every 5 batches
                            torch.cuda.synchronize()

                    # Log progress
                    if batch_idx % (10 * GRADIENT_ACCUMULATION_STEPS) == 0:
                        progress = int((current_step / total_steps) * 100)
                        log_training_progress({
                            'progress': progress,
                            'current_epoch': epoch + 1,
                            'total_epochs': num_epochs,
                            'current_step': current_step,
                            'total_steps': total_steps,
                            'loss': loss.item() * GRADIENT_ACCUMULATION_STEPS,
                            'gpu_memory': f"{torch.cuda.memory_allocated() / 1024**2:.1f}MB" if ACCELERATOR_AVAILABLE else "N/A"
                        })

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if ACCELERATOR_AVAILABLE:
                            torch.cuda.empty_cache()
                        logger.error(f"GPU OOM error at batch {batch_idx}. Skipping batch.")
                        continue
                    else:
                        raise e

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")
            
            # Update training status with validation metrics
            log_training_progress({
                'progress': progress,
                'current_epoch': epoch + 1,
                'total_epochs': num_epochs,
                'loss': loss.item(),
                'val_loss': avg_val_loss
            })
        
        # Save the fine-tuned model
        output_dir = os.path.join(model_path, f"finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Training completed. Duration: {duration}")
        
        training_status.update({
            'progress': 100,
            'is_training': False,
            'end_time': end_time.isoformat()
        })
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Training failed: {error_message}")
        training_status.update({
            'error': error_message,
            'is_training': False,
            'end_time': datetime.now().isoformat()
        })
        if ACCELERATOR_AVAILABLE:
            torch.cuda.empty_cache()

@app.route('/api/models/downloaded', methods=['GET'])
def get_downloaded_models():
    try:
        downloaded = []
        print(f"Checking MODELS_DIR: {MODELS_DIR}")  # Debug print
        
        if not os.path.exists(MODELS_DIR):
            print("Models directory does not exist")  # Debug print
            return jsonify({
                'status': 'success',
                'downloaded_models': []
            })

        for model_dir in os.listdir(MODELS_DIR):
            model_path = os.path.join(MODELS_DIR, model_dir)
            print(f"Checking model path: {model_path}")  # Debug print
            
            if os.path.isdir(model_path):
                # Check for essential model files
                required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
                has_required_files = any(
                    os.path.exists(os.path.join(model_path, file))
                    for file in required_files
                )
                
                if has_required_files:
                    # Convert back from directory name to model ID
                    model_id = model_dir.replace('_', '/')
                    print(f"Found downloaded model: {model_id}")  # Debug print
                    downloaded.append(model_id)
                else:
                    print(f"Missing required files in {model_path}")  # Debug print
        
        print(f"Final downloaded models list: {downloaded}")  # Debug print
        return jsonify({
            'status': 'success',
            'downloaded_models': downloaded
        })
    except Exception as e:
        print(f"Error in get_downloaded_models: {str(e)}")  # Debug print
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Add new route for inference
@app.route('/api/models/inference', methods=['POST'])
def model_inference():
    data = request.json
    model_id = data.get('model_id')
    message = data.get('message')
    use_finetuned = data.get('use_finetuned', False)
    
    if not model_id or not message:
        return jsonify({
            "status": "error",
            "message": "Model ID and message are required"
        }), 400
    
    try:
        # Load model and tokenizer
        safe_dir_name = model_id.replace('/', '_')
        model_path = os.path.join(MODELS_DIR, safe_dir_name)
        
        if use_finetuned:
            # Find the latest finetuned version
            finetuned_versions = [d for d in os.listdir(model_path) 
                                if d.startswith('finetuned_')]
            if not finetuned_versions:
                return jsonify({
                    "status": "error",
                    "message": "No fine-tuned version available"
                }), 404
            
            latest_version = sorted(finetuned_versions)[-1]
            model_path = os.path.join(model_path, latest_version)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto' if ACCELERATOR_AVAILABLE else None,
            torch_dtype=torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Generate response
        inputs = tokenizer(message, return_tensors="pt")
        if ACCELERATOR_AVAILABLE:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            "status": "success",
            "response": response
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/datasets/downloaded', methods=['GET'])
def get_downloaded_datasets():
    try:
        datasets = []
        for filename in os.listdir(DATASETS_DIR):
            file_path = os.path.join(DATASETS_DIR, filename)
            if os.path.isfile(file_path):
                stats = os.stat(file_path)
                datasets.append({
                    'name': filename,
                    'path': file_path,
                    'size': stats.st_size,
                    'uploadedAt': datetime.fromtimestamp(stats.st_mtime).isoformat()
                })
        return jsonify({
            'status': 'success',
            'datasets': datasets
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dataset/structure', methods=['POST'])
def get_dataset_structure():
    data = request.json
    file_path = data.get('file_path')
    
    if not file_path:
        return jsonify({
            'status': 'error',
            'message': 'File path is required'
        }), 400
        
    try:
        file_type = Path(file_path).suffix.lower()
        structure = {}
        
        if file_type in ['.json', '.jsonl']:
            with open(file_path, 'r') as f:
                if file_type == '.jsonl':
                    first_line = f.readline().strip()
                    sample = json.loads(first_line)
                else:
                    data = json.load(f)
                    sample = data[0] if isinstance(data, list) else data
                structure = {
                    'type': 'json',
                    'fields': list(sample.keys())
                }
                
        elif file_type == '.csv':
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                structure = {
                    'type': 'csv',
                    'fields': headers
                }
                
        return jsonify({
            'status': 'success',
            'structure': structure
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dataset/config', methods=['POST'])
def save_dataset_config():
    data = request.json
    file_path = data.get('file_path')
    config = data.get('config')
    
    if not file_path or not config:
        return jsonify({
            'status': 'error',
            'message': 'File path and config are required'
        }), 400
        
    try:
        # Create a safe filename for the config
        safe_name = os.path.basename(file_path) + '.config.json'
        config_path = os.path.join(CONFIG_DIR, safe_name)
        
        # Save the configuration
        with open(config_path, 'w') as f:
            json.dump({
                'file_path': file_path,
                'config': config,
                'configured_at': datetime.now().isoformat()
            }, f)
            
        return jsonify({
            'status': 'success',
            'message': 'Configuration saved'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dataset/config/<path:file_path>', methods=['GET'])
def get_dataset_config(file_path):
    try:
        safe_name = os.path.basename(file_path) + '.config.json'
        config_path = os.path.join(CONFIG_DIR, safe_name)
        
        if not os.path.exists(config_path):
            return jsonify({
                'status': 'error',
                'message': 'No configuration found'
            }), 404
            
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        return jsonify({
            'status': 'success',
            'config': config_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/datasets/configured', methods=['GET'])
def get_configured_datasets():
    try:
        configured = []
        for filename in os.listdir(CONFIG_DIR):
            if filename.endswith('.config.json'):
                with open(os.path.join(CONFIG_DIR, filename), 'r') as f:
                    config_data = json.load(f)
                    configured.append(config_data)
                    
        return jsonify({
            'status': 'success',
            'configured_datasets': configured
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def prepare_training_validation_split(dataset_paths, dataset_configs, validation_split=0.2):
    """Prepare training and validation datasets"""
    all_data = []
    
    for dataset_path in dataset_paths:
        config = dataset_configs.get(dataset_path, {})
        file_type = Path(dataset_path).suffix.lower()
        
        try:
            if file_type in ['.json', '.jsonl']:
                with open(dataset_path, 'r') as f:
                    if file_type == '.jsonl':
                        data = [json.loads(line) for line in f]
                    else:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = [data]
            elif file_type == '.csv':
                data = []
                with open(dataset_path, 'r') as f:
                    reader = csv.DictReader(f)
                    data.extend(list(reader))
            
            # Process data according to config
            processed_data = [{
                'input': str(item[config['inputField']]),
                'output': str(item[config['outputField']])
            } for item in data if config['inputField'] in item and config['outputField'] in item]
            
            all_data.extend(processed_data)
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_path}: {str(e)}")
            raise
    
    if not all_data:
        raise ValueError("No valid data found in the provided datasets")
    
    # Split into training and validation sets
    train_data, val_data = train_test_split(
        all_data, 
        test_size=validation_split,
        random_state=42,
        shuffle=True
    )
    
    logger.info(f"Prepared {len(train_data)} training samples and {len(val_data)} validation samples")
    return train_data, val_data

if __name__ == '__main__':
    app.run(debug=True) 