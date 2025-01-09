# LLM Fine-tuner

A web application for fine-tuning Large Language Models (LLMs) through an intuitive user interface. This application allows users to select models, prepare datasets, and configure fine-tuning parameters through a browser-based interface.

## Features

- Select and download pre-trained language models
- Upload and prepare training datasets
- Configure fine-tuning parameters
- Real-time training progress monitoring
- Validation dataset splitting
- Model inference through chat interface
- Browser-based user interface
- Real-time progress tracking
- Error handling and feedback
- Configurable model selection

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm 6 or higher
- Sufficient disk space for model storage
- GPU acceleration (one of the following):
  - NVIDIA GPU with CUDA support
  - AMD GPU with ROCm support (Linux only)
  - CPU-only (significantly slower)
- Minimum 8GB GPU memory for medium-sized models

## Installation

1. Clone the repository: 
```bash
git clone https://github.com/v912485/llm-finetuner.git
cd llm-finetuner
```

2. Set up the backend:
```bash
cd backend

# Create and activate virtual environment
# On Linux/Mac:
python -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
.\venv\Scripts\activate

# Install backend dependencies based on your GPU:

## For NVIDIA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## For AMD GPU (ROCm, Linux only):
# First install ROCm following instructions at: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

## For CPU only:
pip install torch torchvision torchaudio

# Install other backend requirements
pip install flask flask-cors transformers tqdm scikit-learn
```

3. Set up the frontend:
```bash
cd ../frontend
npm install
```

### Development Tips

- Always activate the virtual environment before running the backend:
  ```bash
  cd backend
  
  # On Linux/Mac:
  source venv/bin/activate
  
  # On Windows:
  .\venv\Scripts\activate
  ```

- To deactivate the virtual environment when you're done:
  ```bash
  deactivate
  ```

- To save your environment requirements:
  ```bash
  pip freeze > requirements.txt
  ```

- To install from requirements.txt:
  ```bash
  pip install -r requirements.txt
  ```

## Configuration

### Model Configuration

Models are configured in `backend/config.json`. The configuration file specifies available models and their requirements:

```json
{
  "models": [
    {
      "id": "model-name",
      "name": "Display Name",
      "size": "small|medium|large",
      "description": "Model description",
      "requirements": {
        "min_gpu_memory": "4GB",
        "recommended_batch_size": 4
      }
    }
  ]
}
```

Add or remove models by editing this configuration file.

## Running the Application

1. Start the backend server:
```bash
cd backend
python app.py
```

2. In a new terminal, start the frontend development server:
```bash
cd frontend
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## Training Process

1. **Model Selection**
   - Choose from available pre-trained models
   - Models are downloaded automatically when selected
   - System checks for GPU memory requirements

2. **Dataset Preparation**
   - Upload training data files
   - Configure input/output field mappings
   - Automatic validation split (configurable percentage)

3. **Training Configuration**
   - Set learning rate
   - Configure batch size
   - Set number of epochs
   - Adjust validation split ratio

4. **Training Monitoring**
   - Real-time progress tracking
   - Loss metrics visualization
   - Validation performance monitoring
   - GPU memory usage tracking

## Data Preparation

### Training Data Format

The application accepts the following file formats:
- JSON (.json)
- JSONL (.jsonl)
- CSV (.csv)
- Text (.txt)

### JSON/JSONL Format
Your training data should be structured as follows:
```json
{
  "instruction": "Classify the sentiment of this text",
  "input": "This movie was absolutely fantastic!",
  "output": "positive"
}
```

### CSV Format
Your CSV should include headers and contain at least these columns:
```csv
instruction,input,output
"Classify the sentiment of this text","This movie was absolutely fantastic!","positive"
```

### Best Practices

1. **Data Cleaning**
   - Remove any duplicate entries
   - Ensure consistent formatting
   - Check for and handle missing values

2. **Data Size**
   - Minimum recommended: 100 examples
   - Optimal range: 1,000-10,000 examples
   - Balance different classes/categories

3. **Quality Control**
   - Verify instruction-output pairs are correct
   - Ensure consistent output format
   - Check for any data leakage

4. **Memory Management**
   - Consider GPU memory limitations
   - Adjust batch size based on model size
   - Use validation split appropriately

## Project Structure

- `backend/`
  - `app.py` - Main Flask application
  - `config.json` - Model configuration
  - `downloaded_models/` - Storage for downloaded models
  - `datasets/` - Storage for uploaded datasets
  - `dataset_configs/` - Dataset configuration storage
  - `logs/` - Training logs

- `frontend/`
  - `src/`
    - `App.js` - Main React component
    - `Chat.js` - Chat interface component
    - `App.css` - Main styles
    - `Chat.css` - Chat interface styles

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with React and Flask
- Uses Hugging Face Transformers library
- PyTorch for model training
- Inspired by the need for accessible LLM fine-tuning tools 

## Memory-Efficient Training

The application supports three training methods:

1. **Full Fine-tuning**
   - Traditional fine-tuning of all model parameters
   - Requires the most GPU memory
   - Best for high-memory GPUs (8GB+)

2. **LoRA (Low-Rank Adaptation)**
   - Fine-tunes low-rank matrices instead of full model
   - Requires ~50% less memory
   - Good for medium-memory GPUs (4-8GB)

3. **QLoRA (Quantized LoRA)**
   - Combines 4-bit quantization with LoRA
   - Requires ~75% less memory
   - Works on low-memory GPUs (2-4GB)

### Additional Dependencies for LoRA/QLoRA

```bash
pip install bitsandbytes peft
``` 

## Environment Setup

For gated models like Gemma, set your Hugging Face token:
```bash
# Linux/Mac:
export HUGGING_FACE_TOKEN="your_token_here"

# Windows:
set HUGGING_FACE_TOKEN=your_token_here
```

Get your token from: https://huggingface.co/settings/tokens