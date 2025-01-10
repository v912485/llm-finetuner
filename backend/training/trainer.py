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
            'device_info': DEVICE_INFO
        }
        self.device = torch.device('cuda' if ACCELERATOR_AVAILABLE else 'cpu')
        self.is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        
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
            self.training_status.update({
                'is_training': True,
                'progress': 0,
                'start_time': datetime.now().isoformat(),
                'model_id': config['model_id'],
                'dataset_info': config['datasets'],
                'total_epochs': config.get('epochs', 3),
                'learning_rate': config.get('learningRate', 0.0001)
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

    def _run_training(self, config):
        try:
            model_id = config['model_id']
            datasets = config['datasets']
            params = config.get('params', {})
            validation_split = float(params.get('validationSplit', 0.2))
            
            logger.info(f"Starting training for model {model_id} with datasets {datasets}")
            logger.info(f"Using validation split: {validation_split}")
            
            # First load model and tokenizer
            safe_model_name = model_id.replace('/', '_')
            model_path = MODELS_DIR / safe_model_name
            
            training_method = params.get('training_method', 'standard')
            logger.info(f"Using training method: {training_method}")
            
            if training_method in ['lora', 'qlora']:
                model_config = self.get_model_config(model_id, use_qlora=(training_method == 'qlora'))
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    **model_config.get('bnb_config', {})
                )
                
                if training_method == 'qlora':
                    model = prepare_model_for_kbit_training(model)
                
                model = get_peft_model(model, model_config['lora_config'])
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Then load and prepare data
            all_data = []
            for dataset_path in datasets:
                with open(dataset_path, 'r') as f:
                    dataset_name = Path(dataset_path).stem
                    config_path = CONFIG_DIR / f"{dataset_name}.config.json"
                    
                    with open(config_path, 'r') as cf:
                        dataset_config = json.load(cf)
                    
                    data = json.load(f)
                    for item in data:
                        all_data.append({
                            'input': item[dataset_config['inputField']],
                            'output': item[dataset_config['outputField']]
                        })
            
            # Shuffle and split data
            random.shuffle(all_data)
            split_idx = int(len(all_data) * (1 - validation_split))
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]
            
            logger.info(f"Dataset split: {len(train_data)} training samples, {len(val_data)} validation samples")
            
            # Create datasets after tokenizer is loaded
            train_dataset = CustomDataset(train_data, tokenizer)
            val_dataset = CustomDataset(val_data, tokenizer)
            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=params.get('batchSize', BATCH_SIZE),
                shuffle=True
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=params.get('batchSize', BATCH_SIZE),
                shuffle=False
            )
            
            # Training setup
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params.get('learningRate', 0.0001)
            )
            
            num_epochs = params.get('epochs', 3)
            total_steps = len(train_dataloader) * num_epochs
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
            
            # Training loop
            model.train()
            best_val_loss = float('inf')
            output_dir = model_path / 'finetuned'
            output_dir.mkdir(exist_ok=True, parents=True)  # Ensure directory exists
            
            logger.info(f"Will save model to: {output_dir}")
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0
                for step, batch in enumerate(train_dataloader):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    
                    loss = outputs.loss
                    train_loss += loss.item()
                    loss.backward()
                    
                    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    # Update progress
                    progress = ((epoch * len(train_dataloader) + step + 1) / (total_steps * num_epochs)) * 100
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
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids
                        )
                        
                        val_loss += outputs.loss.item()
                
                val_loss = val_loss / len(val_dataloader)
                
                # Update status with validation metrics
                self.training_status.update({
                    'current_epoch': epoch + 1,
                    'total_epochs': num_epochs,
                    'val_loss': val_loss
                })
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/(step+1):.4f}, Val Loss: {val_loss:.4f}")
                
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