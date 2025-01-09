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
- CUDA-capable GPU (recommended for training)
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
pip install flask flask-cors torch transformers tqdm scikit-learn
```

3. Set up the frontend:
```bash
cd frontend
npm install
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