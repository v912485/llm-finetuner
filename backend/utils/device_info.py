import torch
import logging
import platform
import os
import psutil

logger = logging.getLogger('training')

def get_device_info():
    """Get detailed information about the available hardware"""
    vm = psutil.virtual_memory()
    device_info = {
        'type': 'cpu',
        'name': 'CPU',
        'backend': 'CPU',
        'memory': f"{vm.total / (1024**3):.1f}GB",
        'memory_total': vm.total,
        'memory_free': vm.available,
        'system': {
            'os': platform.system(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cpu_count': os.cpu_count(),
            'total_memory': f"{vm.total / (1024**3):.1f}GB"
        }
    }
    
    try:
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            # AMD GPU with ROCm
            gpu_props = torch.cuda.get_device_properties(0)
            device_info.update({
                'type': 'cuda',
                'name': 'AMD GPU',
                'backend': 'ROCm',
                'memory': f"{gpu_props.total_memory / (1024**3):.1f}GB",
                'memory_total': gpu_props.total_memory,
                'memory_free': torch.cuda.mem_get_info(0)[0] if torch.cuda.is_available() else 0,
                'device_count': torch.cuda.device_count(),
                'rocm_version': torch.version.hip
            })
            
            # Add details for each GPU
            if torch.cuda.device_count() > 0:
                device_info['devices'] = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    device_info['devices'].append({
                        'index': i,
                        'name': props.name,
                        'memory': f"{props.total_memory / (1024**3):.1f}GB",
                        'memory_total': props.total_memory,
                        'memory_free': free_mem,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
                    
        elif torch.cuda.is_available():
            # NVIDIA GPU with CUDA
            gpu_props = torch.cuda.get_device_properties(0)
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            device_info.update({
                'type': 'cuda',
                'name': torch.cuda.get_device_name(0),
                'backend': 'CUDA',
                'memory': f"{gpu_props.total_memory / (1024**3):.1f}GB",
                'memory_total': gpu_props.total_memory,
                'memory_free': free_mem,
                'device_count': torch.cuda.device_count(),
                'cuda_version': torch.version.cuda
            })
            
            # Add details for each GPU
            if torch.cuda.device_count() > 0:
                device_info['devices'] = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    f_mem, t_mem = torch.cuda.mem_get_info(i)
                    device_info['devices'].append({
                        'index': i,
                        'name': props.name,
                        'memory': f"{props.total_memory / (1024**3):.1f}GB",
                        'memory_total': props.total_memory,
                        'memory_free': f_mem,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
    except Exception as e:
        logger.error(f"Error getting GPU information: {str(e)}")
        device_info['error'] = str(e)
        
    return device_info

# Initialize device info
DEVICE_INFO = get_device_info()
ACCELERATOR_AVAILABLE = DEVICE_INFO['type'] == 'cuda' 

# Log device information
if ACCELERATOR_AVAILABLE:
    logger.info(f"PyTorch detected GPU: Using {DEVICE_INFO['backend']} on {DEVICE_INFO['name']} for potential acceleration (e.g., training).")
    if 'device_count' in DEVICE_INFO and DEVICE_INFO['device_count'] > 0:
        logger.info(f"Found {DEVICE_INFO['device_count']} GPU(s)")
        for i, device in enumerate(DEVICE_INFO.get('devices', [])):
            logger.info(f"  GPU {i}: {device['name']} with {device['memory']} memory")
else:
    logger.warning(f"PyTorch did not detect GPU acceleration. Training will use CPU.")
    logger.warning(f"Note: SGLang might still utilize GPU for inference if properly configured, independent of this PyTorch check.")
    logger.info(f"System: {DEVICE_INFO['system']['os']}, Python {DEVICE_INFO['system']['python_version']}, PyTorch {DEVICE_INFO['system']['torch_version']}")
    logger.info(f"CPU cores: {DEVICE_INFO['system']['cpu_count']}, Total memory: {DEVICE_INFO['system']['total_memory']}")