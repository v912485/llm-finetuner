import torch
import logging

logger = logging.getLogger('training')

def get_device_info():
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return {
            'type': 'cuda',
            'name': 'AMD GPU',
            'backend': 'ROCm',
            'memory': f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB"
        }
    elif torch.cuda.is_available():
        return {
            'type': 'cuda',
            'name': torch.cuda.get_device_name(0),
            'backend': 'CUDA',
            'memory': f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB"
        }
    else:
        return {
            'type': 'cpu',
            'name': 'CPU',
            'backend': 'CPU',
            'memory': 'N/A'
        }

# Initialize device info
DEVICE_INFO = get_device_info()
ACCELERATOR_AVAILABLE = DEVICE_INFO['type'] == 'cuda' 

if ACCELERATOR_AVAILABLE:
    logger.info(f"Using {DEVICE_INFO['backend']} on {DEVICE_INFO['name']} with {torch.version.cuda}")
else:
    logger.warning(f"No GPU acceleration available on torch version {torch.version.cuda}. Using CPU for training.")