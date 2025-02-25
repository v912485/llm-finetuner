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
SAVED_MODELS_DIR = BASE_DIR / "saved_models"

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)
SAVED_MODELS_DIR.mkdir(exist_ok=True)

# Training settings - allow override from environment variables
GRADIENT_ACCUMULATION_STEPS = int(os.getenv('GRADIENT_ACCUMULATION_STEPS', 8))
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 128))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 3))

# Environment variables
HF_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
if not HF_TOKEN:
    logging.warning("HUGGING_FACE_TOKEN not found in environment variables")

# Log level configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = BASE_DIR / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Set up file handler
    log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_LEVELS.get(LOG_LEVEL, logging.INFO))
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVELS.get(LOG_LEVEL, logging.INFO))
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Set up logger
    logger = logging.getLogger('training')
    logger.setLevel(LOG_LEVELS.get(LOG_LEVEL, logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log startup information
    logger.info(f"Starting application with settings:")
    logger.info(f"- BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"- MAX_LENGTH: {MAX_LENGTH}")
    logger.info(f"- GRADIENT_ACCUMULATION_STEPS: {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"- LOG_LEVEL: {LOG_LEVEL}")
    
    return logger 