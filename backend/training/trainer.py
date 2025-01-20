import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import BitsAndBytesConfig
from training.dataset import CustomDataset
from utils.device_info import DEVICE_INFO, ACCELERATOR_AVAILABLE
from config.settings import MODELS_DIR, GRADIENT_ACCUMULATION_STEPS, MAX_LENGTH, BATCH_SIZE, CONFIG_DIR
import logging
from datetime import datetime
import os
import json
from pathlib import Path
import random

logger = logging.getLogger('training')

class Trainer:
    def __init__(self):
        self.training_status = {
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
            'learning_rate': None,
            'device_info': DEVICE_INFO,
            'history': []  # Add history array to track metrics
        }
        self.device = torch.device('cuda' if ACCELERATOR_AVAILABLE else 'cpu')
        self.is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        self.should_stop = False  # Add flag for cancellation
        
        # Log device information at initialization
        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Device info: {DEVICE_INFO}")
        if ACCELERATOR_AVAILABLE:
            logger.info(f"GPU Memory: {DEVICE_INFO['memory']}")
            if self.is_rocm:
                logger.info("Using AMD ROCm backend")
            else:
                logger.info("Using NVIDIA CUDA backend")
        else:
            logger.warning("No GPU acceleration available, using CPU")

    def is_training(self):
        return self.training_status['is_training']

    def get_status(self):
        return self.training_status

    def get_model_config(self, model_id, use_qlora=False):
        target_modules = {
            'gpt2': ["c_attn"],
            'facebook/opt': ["q_proj", "v_proj"],
            'google/gemma': ["q_proj", "v_proj"],
            'default': ["query", "value"]
        }

        model_prefix = next(
            (prefix for prefix in target_modules.keys() if model_id.startswith(prefix)),
            'default'
        )
        modules = target_modules[model_prefix]

        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32
            )
            
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            return {"bnb_config": bnb_config, "lora_config": lora_config}
        else:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            return {"lora_config": lora_config}

    def start_training(self, config):
        try:
            # Get epochs from params if provided, otherwise use default
            num_epochs = config.get('params', {}).get('epochs', 3)
            
            self.training_status.update({
                'is_training': True,
                'progress': 0,
                'start_time': datetime.now().isoformat(),
                'model_id': config['model_id'],
                'dataset_info': config['datasets'],
                'current_epoch': 1,  # Start at 1 instead of 0
                'total_epochs': num_epochs,
                'learning_rate': config.get('params', {}).get('learningRate', 0.0001)
            })

            # Start training in a separate thread
            import threading
            thread = threading.Thread(target=self._run_training, args=(config,))
            thread.start()

        except Exception as e:
            self.training_status.update({
                'is_training': False,
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            raise

    def cancel_training(self):
        """Cancel ongoing training"""
        logger.info("Cancelling training...")
        self.should_stop = True
        self.training_status.update({
            'is_training': False,
            'end_time': datetime.now().isoformat(),
            'error': 'Training cancelled by user'
        })

    def _load_and_validate_dataset(self, dataset_path, config):
        """Helper function to load and validate dataset with error handling"""
        try:
            # First try to load the entire file
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                    logger.info(f"Successfully loaded dataset: {dataset_path}")
                    return dataset
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON load failed, trying line-by-line: {str(e)}")
                
                # If that fails, try to read line by line
                valid_entries = []
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        try:
                            entry = json.loads(line)
                            valid_entries.append(entry)
                        except json.JSONDecodeError as line_error:
                            logger.error(f"Error parsing line {line_num}: {str(line_error)}")
                            continue
                
                if not valid_entries:
                    raise ValueError("No valid JSON entries found in file")
                    
                logger.info(f"Loaded {len(valid_entries)} valid entries from {dataset_path}")
                return valid_entries
                
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_path}: {str(e)}")
            raise

    def _run_training(self, config):
        try:
            self.should_stop = False  # Reset stop flag
            model_id = config['model_id']
            datasets = config['datasets']
            params = config.get('params', {})
            
            # Force single GPU if specified
            if params.get('force_single_gpu'):
                gpu_index = params.get('gpu_index', 0)
                if torch.cuda.is_available():
                    torch.cuda.set_device(gpu_index)
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
                    self.device = torch.device(f'cuda:{gpu_index}')
                    logger.info(f"Forcing use of GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Get training parameters with defaults
            validation_split = float(params.get('validationSplit', 0.2))
            num_epochs = int(params.get('epochs', 3))
            batch_size = int(params.get('batchSize', BATCH_SIZE))
            learning_rate = float(params.get('learningRate', 0.0001))
            training_method = params.get('training_method', 'standard')
            
            logger.info(f"Training parameters:")
            logger.info(f"- Device: {self.device}")
            logger.info(f"- Epochs: {num_epochs}")
            logger.info(f"- Batch size: {batch_size}")
            logger.info(f"- Learning rate: {learning_rate}")
            logger.info(f"- Validation split: {validation_split}")
            logger.info(f"- Training method: {training_method}")
            
            # First load model and tokenizer
            safe_model_name = model_id.replace('/', '_')
            model_path = MODELS_DIR / safe_model_name
            
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Load model based on training method
            if training_method in ['lora', 'qlora']:
                model_config = self.get_model_config(model_id, use_qlora=(training_method == 'qlora'))
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=None,  # Disable auto device mapping
                    trust_remote_code=True,
                    **model_config.get('bnb_config', {})
                )
                
                if training_method == 'qlora':
                    model = prepare_model_for_kbit_training(model)
                
                model = get_peft_model(model, model_config['lora_config'])
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=None,  # Disable auto device mapping
                    trust_remote_code=True
                )
            
            # Explicitly move model to device
            model = model.to(self.device)
            
            # Enable gradient checkpointing if available
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            # Disable model parallelism
            if hasattr(model, 'is_parallelizable'):
                model.is_parallelizable = False
                model.model_parallel = False
            
            # Load dataset configurations
            all_data = []
            for dataset_path in datasets:
                dataset_name = Path(dataset_path).stem
                config_path = CONFIG_DIR / f"{dataset_name}.config.json"
                
                logger.info(f"Loading config from: {config_path}")
                
                try:
                    with open(config_path, 'r') as cf:
                        dataset_config = json.load(cf)
                        
                    # Load and validate dataset
                    dataset = self._load_and_validate_dataset(dataset_path, dataset_config)
                    
                    input_field = dataset_config['input_field']
                    output_field = dataset_config['output_field']
                    
                    # Process dataset entries
                    for entry in dataset:
                        if isinstance(entry, dict) and input_field in entry and output_field in entry:
                            all_data.append({
                                'input': str(entry[input_field]),
                                'output': str(entry[output_field])
                            })
                        else:
                            logger.warning(f"Skipping invalid entry: {entry}")
                            
                    logger.info(f"Processed {len(all_data)} valid entries from {dataset_path}")
                            
                except Exception as e:
                    logger.error(f"Error processing dataset {dataset_path}: {str(e)}")
                    raise
                
            if not all_data:
                raise ValueError("No valid training data found in datasets")
                
            logger.info(f"Total valid entries for training: {len(all_data)}")
            
            # Shuffle and split data
            random.shuffle(all_data)
            split_idx = int(len(all_data) * (1 - validation_split))
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]
            
            logger.info(f"Dataset split: {len(train_data)} training samples, {len(val_data)} validation samples")
            
            # Create datasets with loaded tokenizer
            train_dataset = CustomDataset(train_data, tokenizer)
            val_dataset = CustomDataset(val_data, tokenizer)
            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # Training setup
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate
            )
            
            total_training_steps = len(train_dataloader) * num_epochs
            total_validation_steps = len(val_dataloader) * num_epochs
            total_steps = total_training_steps + total_validation_steps
            current_step = 0
            
            logger.info(f"Total steps: {total_steps} (Training: {total_training_steps}, Validation: {total_validation_steps})")
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
            
            # Initialize history at the start
            self.training_status['history'] = []
            
            # Training loop
            model.train()
            best_val_loss = float('inf')
            output_dir = model_path / 'finetuned'
            output_dir.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"Will save model to: {output_dir}")
            logger.info(f"Using device: {self.device}")
            
            for epoch in range(num_epochs):
                if self.should_stop:
                    logger.info("Training cancelled")
                    return

                # Training phase
                model.train()
                train_loss = 0
                
                # Update current epoch at the start of each epoch (1-based indexing)
                self.training_status.update({
                    'current_epoch': epoch + 1,
                    'total_epochs': num_epochs
                })
                
                for step, batch in enumerate(train_dataloader):
                    if self.should_stop:
                        logger.info("Training cancelled")
                        return

                    # Move batch to device and ensure it stays there
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    labels = batch['input_ids'].clone()
                    
                    # Clear CUDA cache periodically
                    if step % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    train_loss += loss.item()
                    loss.backward()
                    
                    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    current_step += 1
                    # Update progress based on total steps
                    progress = (current_step / total_steps) * 100
                    
                    self.training_status.update({
                        'progress': round(progress, 2),
                        'loss': loss.item(),
                        'current_step': step + 1,
                        'total_steps': len(train_dataloader),
                        'train_loss': train_loss / (step + 1)
                    })
                
                # Validation phase
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        if self.should_stop:
                            logger.info("Training cancelled")
                            return

                        # Move batch to device and ensure it stays there
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        labels = batch['input_ids'].clone()
                        
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=labels
                        )
                        
                        val_loss += outputs.loss.item()
                        current_step += 1
                        progress = (current_step / total_steps) * 100
                        self.training_status['progress'] = round(progress, 2)
                
                val_loss = val_loss / len(val_dataloader)
                epoch_train_loss = train_loss / len(train_dataloader)
                
                # Add metrics to history
                epoch_metrics = {
                    'epoch': epoch + 1,  # Use 1-based epoch numbering
                    'train_loss': epoch_train_loss,
                    'val_loss': val_loss,
                    'timestamp': datetime.now().isoformat()
                }
                self.training_status['history'].append(epoch_metrics)
                
                # Update status with epoch completion
                self.training_status.update({
                    'current_epoch': epoch + 1,  # Use 1-based epoch numbering
                    'total_epochs': num_epochs,
                    'val_loss': val_loss,
                    'train_loss': epoch_train_loss,
                    'current_metrics': epoch_metrics
                })
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"New best validation loss: {val_loss:.4f}. Saving model...")
                    
                    try:
                        if training_method in ['lora', 'qlora']:
                            model.save_pretrained(output_dir)
                            logger.info("Saved LoRA model weights")
                        else:
                            model.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            logger.info("Saved full model and tokenizer")
                        
                        # Save training configuration
                        training_config = {
                            'model_id': model_id,
                            'training_method': training_method,
                            'best_val_loss': best_val_loss,
                            'epochs_completed': epoch + 1,
                            'training_params': params,
                            'save_time': datetime.now().isoformat()
                        }
                        
                        with open(output_dir / 'training_config.json', 'w') as f:
                            json.dump(training_config, f, indent=2)
                            
                        logger.info(f"Saved training configuration to {output_dir / 'training_config.json'}")
                        
                    except Exception as save_error:
                        logger.error(f"Error saving model: {str(save_error)}")
                        raise
            
            # Save final model state if it's the best one
            if val_loss <= best_val_loss:
                logger.info("Final model is the best model. Saving...")
                if training_method in ['lora', 'qlora']:
                    model.save_pretrained(output_dir)
                else:
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
            
            self.training_status.update({
                'is_training': False,
                'progress': 100,
                'end_time': datetime.now().isoformat(),
                'final_val_loss': val_loss,
                'best_val_loss': best_val_loss
            })
            
            logger.info(f"Training completed. Model saved at: {output_dir}")
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            self.training_status.update({
                'is_training': False,
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            raise 