# LLM Fine-tuner

A web application for fine-tuning Large Language Models (LLMs) through an intuitive user interface. This application allows users to select models, prepare datasets, and configure fine-tuning parameters through a browser-based interface.

## Features

- Select and download pre-trained language models
- Upload and prepare training datasets
- Configure fine-tuning parameters
- Browser-based user interface
- Real-time download progress tracking
- Error handling and feedback

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm 6 or higher
- Sufficient disk space for model storage
- CUDA-capable GPU (recommended for training)

## Installation

1. Clone the repository: 
```bash
git clone https://github.com/yourusername/llm-finetuner.git
cd llm-finetuner
```

2. Set up the backend:
```bash
cd backend
pip install flask flask-cors torch transformers tqdm
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

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

### Text Format
For plain text files, each training example should be separated by newlines and follow this format:
```text
### Instruction: Classify the sentiment of this text
### Input: This movie was absolutely fantastic!
### Output: positive
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

## Development

- Backend API endpoints are defined in `backend/app.py`
- Frontend React components are in `frontend/src/`
- Styles are managed in `frontend/src/App.css`
- Model downloads are stored in `backend/downloaded_models/`
- Datasets are stored in `backend/datasets/`

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
- Inspired by the need for accessible LLM fine-tuning tools 