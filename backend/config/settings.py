import os
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory configuration
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "downloaded_models"
DATASETS_DIR = BASE_DIR / "datasets"
CONFIG_DIR = BASE_DIR / "dataset_configs"

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

# Training settings
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 128
BATCH_SIZE = 8

# Add SAVED_MODELS_DIR to settings
SAVED_MODELS_DIR = MODELS_DIR / 'saved_models'

# Environment variables
HF_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
if not HF_TOKEN:
    logging.warning("HUGGING_FACE_TOKEN not found in environment variables")

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = BASE_DIR / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Set up file handler
    log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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