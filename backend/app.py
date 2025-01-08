from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm
import sys

app = Flask(__name__)
CORS(app)

MODELS_DIR = "downloaded_models"
DATASETS_DIR = "datasets"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

@app.route('/api/models/available', methods=['GET'])
def get_available_models():
    # List of supported models for fine-tuning
    models = [
        {"id": "gpt2", "name": "GPT-2", "size": "small"},
        {"id": "gpt2-medium", "name": "GPT-2 Medium", "size": "medium"},
        {"id": "facebook/opt-350m", "name": "OPT 350M", "size": "medium"}
    ]
    return jsonify(models)

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
        # Add progress callback
        def progress_callback(evolution):
            sys.stdout.write(f'\rDownloading model: {evolution["progress"]:.2f}%')
            sys.stdout.flush()
        
        # Download model and tokenizer with progress tracking
        print(f"Downloading model: {model_id}")
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
        model_path = os.path.join(MODELS_DIR, model_id.replace('/', '_'))
        print(f"Saving model to: {model_path}")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
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
        
        # Process the dataset (implement your processing logic here)
        
        return jsonify({
            "status": "success",
            "message": "Dataset prepared successfully",
            "dataset_path": dataset_path
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 