import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from training.dataset import CustomDataset
from utils.device_info import DEVICE_INFO, ACCELERATOR_AVAILABLE
from config.settings import MODELS_DIR, GRADIENT_ACCUMULATION_STEPS, MAX_LENGTH, BATCH_SIZE, CONFIG_DIR, DATASETS_DIR
import logging
from datetime import datetime
import os
import json
from pathlib import Path
import random
import uuid
from utils.datasets import resolve_dataset_path, load_json_dataset
import threading
import importlib

logger = logging.getLogger('training')

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    TaskType = None

class Trainer:
    def __init__(self):
        self.training_status = {
            'is_training': False,
            'is_cancelling': False,
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
            'history': [],  # Add history array to track metrics
            'queue_length': 0,
            'queue_position': None,
            'queued_runs': [],
            'active_run_id': None,
            'current_run_id': None
        }
        self.device = torch.device('cuda' if ACCELERATOR_AVAILABLE else 'cpu')
        self.is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        self._cancel_event = threading.Event()
        self._training_thread = None
        self._queue = []
        self._queue_lock = threading.Lock()
        self._queue_thread = None
        self._active_run_id = None
        
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
        # Update dynamic device info (like free memory) if possible
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    if 'devices' in self.training_status['device_info'] and i < len(self.training_status['device_info']['devices']):
                        self.training_status['device_info']['devices'][i]['memory_free'] = free_mem
                    if i == 0:
                        self.training_status['device_info']['memory_free'] = free_mem
            except Exception:
                pass
        else:
            # Update CPU memory info
            try:
                import psutil
                vm = psutil.virtual_memory()
                self.training_status['device_info']['memory_free'] = vm.available
            except Exception:
                pass
        return self.training_status

    def _generate_run_id(self):
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"{timestamp}_{uuid.uuid4().hex[:8]}"

    def _get_run_dir(self, model_id, run_id):
        safe_model_name = model_id.replace('/', '_')
        return MODELS_DIR / safe_model_name / 'finetuned' / 'runs' / run_id

    def _write_run_metadata(self, run_dir, payload):
        run_dir.mkdir(parents=True, exist_ok=True)
        temp_path = run_dir / 'run.json.tmp'
        final_path = run_dir / 'run.json'
        with open(temp_path, 'w') as f:
            json.dump(payload, f, indent=2)
        temp_path.replace(final_path)

    def _update_run_metadata(self, run_dir, updates):
        run_path = run_dir / 'run.json'
        payload = {}
        if run_path.exists():
            try:
                with open(run_path, 'r') as f:
                    payload = json.load(f)
            except Exception:
                payload = {}
        payload.update(updates)
        self._write_run_metadata(run_dir, payload)

    def _append_metrics(self, run_dir, metrics):
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / 'metrics.jsonl'
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + "\n")

    def _load_metrics_history(self, run_dir):
        metrics_path = run_dir / 'metrics.jsonl'
        history = []
        if not metrics_path.exists():
            return history
        with open(metrics_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return history

    def _resolve_eval_hooks(self, hook_specs):
        hooks = []
        for spec in hook_specs:
            if not spec:
                continue
            module_path = None
            func_name = None
            if isinstance(spec, dict):
                module_path = spec.get('module')
                func_name = spec.get('function')
            elif isinstance(spec, str):
                if ':' in spec:
                    module_path, func_name = spec.split(':', 1)
                elif '.' in spec:
                    module_path, func_name = spec.rsplit('.', 1)
            if not module_path or not func_name:
                logger.warning(f"Invalid eval hook spec: {spec}")
                continue
            try:
                module = importlib.import_module(module_path)
                hook = getattr(module, func_name, None)
                if not callable(hook):
                    logger.warning(f"Eval hook not callable: {module_path}.{func_name}")
                    continue
                hooks.append((f"{module_path}.{func_name}", hook))
            except Exception as e:
                logger.warning(f"Failed to load eval hook {spec}: {e}")
        return hooks

    def _apply_eval_hooks(self, hooks, context):
        results = {}
        for name, hook in hooks:
            try:
                payload = hook(context)
                if isinstance(payload, dict):
                    results.update(payload)
                elif payload is not None:
                    results[name] = payload
            except Exception as e:
                logger.warning(f"Eval hook {name} failed: {e}")
        return results

    def _save_checkpoint(self, checkpoint_path, state):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = checkpoint_path.with_suffix('.tmp')
        torch.save(state, temp_path)
        temp_path.replace(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        if not checkpoint_path.exists():
            return None
        return torch.load(checkpoint_path, map_location=self.device)

    def _update_queue_status_locked(self):
        queued_runs = [
            {
                'run_id': job['run_id'],
                'model_id': job['config'].get('model_id'),
                'datasets': job['config'].get('datasets')
            }
            for job in self._queue
        ]
        self.training_status.update({
            'queue_length': len(self._queue),
            'queued_runs': queued_runs,
            'active_run_id': self._active_run_id
        })

    def _enqueue_training(self, config):
        params = config.get('params', {})
        run_id = config.get('run_id') or params.get('resume_run_id') or params.get('resume_from_run_id')
        if not run_id:
            run_id = self._generate_run_id()
        config['run_id'] = run_id

        job = {
            'run_id': run_id,
            'config': config,
            'queued_at': datetime.now().isoformat()
        }
        with self._queue_lock:
            self._queue.append(job)
            self._update_queue_status_locked()
            queue_position = len(self._queue)

        self.training_status.update({
            'queue_position': queue_position,
            'current_run_id': run_id
        })
        return run_id

    def _start_queue_worker_if_needed(self):
        with self._queue_lock:
            if self._queue_thread and self._queue_thread.is_alive():
                return
            self._queue_thread = threading.Thread(target=self._process_queue, daemon=True)
            self._queue_thread.start()

    def _process_queue(self):
        while True:
            with self._queue_lock:
                if not self._queue:
                    self._update_queue_status_locked()
                    return
                job = self._queue.pop(0)
                self._active_run_id = job['run_id']
                self._update_queue_status_locked()
                self.training_status['queue_position'] = None

            self._run_training(job['config'])

            with self._queue_lock:
                self._active_run_id = None
                self._update_queue_status_locked()

    def get_model_config(self, model_id, use_qlora=False):
        if LoraConfig is None or TaskType is None:
            raise RuntimeError("peft is not installed; LoRA/QLoRA training is unavailable on this server.")

        # Modern LLMs (Llama, Qwen, Mistral, Gemma, etc.) usually use q_proj, v_proj
        # GPT-2 uses c_attn
        # Older models might use query, value
        
        model_id_lower = model_id.lower()
        
        if 'gpt2' in model_id_lower:
            modules = ["c_attn"]
        elif any(x in model_id_lower for x in ['llama', 'qwen', 'gemma', 'opt', 'mistral', 'deepseek', 'phi']):
            modules = ["q_proj", "v_proj"]
        else:
            # Default to q_proj, v_proj as it's standard for modern causal LMs
            modules = ["q_proj", "v_proj"]
            
        logger.info(f"Selected target modules {modules} for model {model_id}")

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
            run_id = self._enqueue_training(config)
            self._start_queue_worker_if_needed()
            return run_id

        except Exception as e:
            self.training_status.update({
                'is_training': False,
                'is_cancelling': False,
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            raise

    def cancel_training(self):
        """Cancel ongoing training"""
        logger.info("Cancelling training...")
        self._cancel_event.set()
        self.training_status.update({
            'is_cancelling': True,
            'error': 'Cancellation requested'
        })

    def _load_and_validate_dataset(self, dataset_path, config):
        """Helper function to load and validate dataset with error handling"""
        try:
            dataset = load_json_dataset(Path(dataset_path))
            logger.info(f"Successfully loaded dataset: {dataset_path} ({len(dataset)} records)")
            return dataset
                
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_path}: {str(e)}")
            raise

    def _run_training(self, config):
        try:
            self._cancel_event.clear()
            torch.set_grad_enabled(True)
            model_id = config['model_id']
            dataset_ids = config['datasets']
            params = config.get('params', {})
            run_id = config.get('run_id') or self._generate_run_id()
            config['run_id'] = run_id
            
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
            max_length = int(params.get('maxLength', MAX_LENGTH))
            grad_accum_steps = int(params.get('gradientAccumulationSteps', GRADIENT_ACCUMULATION_STEPS))
            checkpoint_enabled = bool(params.get('checkpoint_enabled', True))
            checkpoint_interval_epochs = int(params.get('checkpoint_interval_epochs', 1))
            if checkpoint_interval_epochs < 1:
                checkpoint_interval_epochs = 1
            seed = params.get('seed')
            if seed is None:
                seed = random.randint(0, 2**32 - 1)

            eval_metric_names = params.get('eval_metrics')
            if eval_metric_names is None:
                eval_metric_names = ['accuracy', 'bleu', 'rouge']
            eval_metric_set = {str(name).lower() for name in eval_metric_names}
            compute_accuracy = 'accuracy' in eval_metric_set or 'token_accuracy' in eval_metric_set
            compute_bleu = 'bleu' in eval_metric_set
            compute_rouge = 'rouge' in eval_metric_set or any(
                name in eval_metric_set for name in ['rouge1', 'rouge2', 'rougel']
            )
            eval_hooks = self._resolve_eval_hooks(params.get('eval_hooks', []))
            
            logger.info(f"Training parameters:")
            logger.info(f"- Device: {self.device}")
            logger.info(f"- Epochs: {num_epochs}")
            logger.info(f"- Batch size: {batch_size}")
            logger.info(f"- Learning rate: {learning_rate}")
            logger.info(f"- Validation split: {validation_split}")
            logger.info(f"- Training method: {training_method}")
            logger.info(f"- Max length: {max_length}")
            logger.info(f"- Gradient accumulation steps: {grad_accum_steps}")
            logger.info(f"- Checkpointing enabled: {checkpoint_enabled}")
            if checkpoint_enabled:
                logger.info(f"- Checkpoint interval (epochs): {checkpoint_interval_epochs}")
            logger.info(f"- Eval metrics: {sorted(eval_metric_set) if eval_metric_set else 'none'}")
            if eval_hooks:
                logger.info(f"- Eval hooks: {[name for name, _ in eval_hooks]}")
            resume_from_run_id = params.get('resume_run_id') or params.get('resume_from_run_id')
            resume_from_checkpoint_path = params.get('resume_checkpoint_path')
            restart_from_run_id = params.get('restart_from_run_id')
            restart_from_checkpoint_path = params.get('restart_from_checkpoint_path')
            resume_enabled = bool(params.get('resume_from_checkpoint', True))

            run_dir = self._get_run_dir(model_id, run_id)
            checkpoint_path = run_dir / 'checkpoint.pt'

            if resume_from_run_id and run_id != resume_from_run_id:
                run_id = resume_from_run_id
                config['run_id'] = run_id
                run_dir = self._get_run_dir(model_id, run_id)
                checkpoint_path = run_dir / 'checkpoint.pt'

            history = []
            if resume_enabled and (run_dir / 'metrics.jsonl').exists():
                history = self._load_metrics_history(run_dir)

            run_metadata = {
                'run_id': run_id,
                'model_id': model_id,
                'datasets': dataset_ids,
                'params': params,
                'seed': seed,
                'created_at': datetime.now().isoformat(),
                'resume_from_run_id': resume_from_run_id,
                'resume_from_checkpoint_path': resume_from_checkpoint_path,
                'restart_from_run_id': restart_from_run_id,
                'restart_from_checkpoint_path': restart_from_checkpoint_path
            }
            self._update_run_metadata(run_dir, run_metadata)

            self.training_status.update({
                'is_training': True,
                'is_cancelling': False,
                'progress': 0,
                'error': None,
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'model_id': model_id,
                'dataset_info': dataset_ids,
                'current_epoch': 1,
                'total_epochs': num_epochs,
                'learning_rate': learning_rate,
                'loss': None,
                'train_loss': None,
                'val_loss': None,
                'final_val_loss': None,
                'best_val_loss': None,
                'current_metrics': None,
                'current_step': None,
                'total_steps': None,
                'history': history,
                'current_run_id': run_id,
                'active_run_id': run_id
            })

            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            # First load model and tokenizer
            safe_model_name = model_id.replace('/', '_')
            model_path = MODELS_DIR / safe_model_name
            
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Load model based on training method
            common_model_kwargs = {
                "device_map": None,  # Disable auto device mapping
                "trust_remote_code": True,
            }
            if "gemma" in (model_id or "").lower():
                common_model_kwargs["attn_implementation"] = "eager"

            if training_method in ['lora', 'qlora']:
                if get_peft_model is None:
                    raise RuntimeError("peft is not installed; LoRA/QLoRA training is unavailable on this server.")
                model_config = self.get_model_config(model_id, use_qlora=(training_method == 'qlora'))
                
                if training_method == 'qlora':
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=model_config.get('bnb_config'),
                        **common_model_kwargs
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **common_model_kwargs
                    )
                
                if training_method == 'qlora':
                    if prepare_model_for_kbit_training is None:
                        raise RuntimeError("peft is not installed; QLoRA training is unavailable on this server.")
                    model = prepare_model_for_kbit_training(model)
                
                model = get_peft_model(model, model_config['lora_config'])
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **common_model_kwargs
                )
            
            # Explicitly move model to device
            model = model.to(self.device)
            
            # Enable gradient checkpointing if available
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                if getattr(model, "config", None) is not None:
                    model.config.use_cache = False
                    if hasattr(model.config, "gradient_checkpointing"):
                        model.config.gradient_checkpointing = True

            if training_method == 'standard':
                for p in model.parameters():
                    p.requires_grad_(True)

            if not any(p.requires_grad for p in model.parameters()):
                raise RuntimeError("No trainable parameters found; model appears to be fully frozen.")
            
            # Disable model parallelism
            if hasattr(model, 'is_parallelizable'):
                model.is_parallelizable = False
                model.model_parallel = False
            
            # Load dataset configurations
            all_data = []
            dataset_configs_list = []
            for dataset_id in dataset_ids:
                config_path = CONFIG_DIR / f"{dataset_id}.config.json"
                
                logger.info(f"Loading config from: {config_path}")
                
                try:
                    with open(config_path, 'r') as cf:
                        dataset_config = json.load(cf)
                        
                    dataset_path = resolve_dataset_path(DATASETS_DIR, dataset_id)
                    # Load and validate dataset
                    dataset = self._load_and_validate_dataset(dataset_path, dataset_config)
                    
                    input_field = dataset_config['input_field']
                    output_field = dataset_config['output_field']
                    dataset_configs_list.append((input_field, output_field))
                    
                    # Process dataset entries - just pass them through
                    # CustomDataset will handle field extraction
                    for entry in dataset:
                        if isinstance(entry, dict):
                            all_data.append(entry)
                        else:
                            logger.warning(f"Skipping invalid entry: {entry}")
                            
                    logger.info(f"Processed {len(all_data)} valid entries from dataset_id={dataset_id}")
                            
                except Exception as e:
                    logger.error(f"Error processing dataset dataset_id={dataset_id}: {str(e)}")
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
            
            # Use first dataset's config for field names (all datasets should use same schema)
            input_field, output_field = dataset_configs_list[0] if dataset_configs_list else ('input', 'output')
            logger.info(f"Using field mapping: input_field={input_field}, output_field={output_field}")
            
            # Create datasets with loaded tokenizer
            train_dataset = CustomDataset(train_data, tokenizer, input_field, output_field)
            val_dataset = CustomDataset(val_data, tokenizer, input_field, output_field)
            
            # Update max_length if provided
            train_dataset.max_length = max_length
            val_dataset.max_length = max_length
            
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

            start_epoch = 0
            best_val_loss = float('inf')
            checkpoint_source_path = None

            if restart_from_checkpoint_path:
                checkpoint_source_path = Path(restart_from_checkpoint_path)
            elif restart_from_run_id:
                checkpoint_source_path = self._get_run_dir(model_id, restart_from_run_id) / 'checkpoint.pt'
            elif resume_enabled:
                if resume_from_checkpoint_path:
                    checkpoint_source_path = Path(resume_from_checkpoint_path)
                else:
                    checkpoint_source_path = checkpoint_path

            if checkpoint_source_path:
                checkpoint = self._load_checkpoint(checkpoint_source_path)
                if checkpoint:
                    model_state = checkpoint.get('model_state') or checkpoint.get('model_state_dict')
                    if model_state:
                        model.load_state_dict(model_state)
                    else:
                        logger.warning("Checkpoint missing model state; continuing without loading weights")
                    optimizer_state = checkpoint.get('optimizer_state')
                    if optimizer_state:
                        optimizer.load_state_dict(optimizer_state)
                    scheduler_state = checkpoint.get('scheduler_state')
                    if scheduler_state:
                        scheduler.load_state_dict(scheduler_state)
                    start_epoch = int(checkpoint.get('epoch', 0))
                    current_step = int(checkpoint.get('global_step', 0))
                    best_val_loss = checkpoint.get('best_val_loss', best_val_loss)

                    rng_state = checkpoint.get('rng_state')
                    if rng_state is not None:
                        torch.set_rng_state(rng_state)
                    cuda_rng_state = checkpoint.get('cuda_rng_state')
                    if cuda_rng_state is not None and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(cuda_rng_state)

                    logger.info(f"Resuming from checkpoint: {checkpoint_source_path}")
                else:
                    logger.info(f"No checkpoint found at: {checkpoint_source_path}")

            if history:
                self.training_status['history'] = history
            if start_epoch > 0:
                self.training_status.update({
                    'current_epoch': start_epoch + 1,
                    'best_val_loss': best_val_loss
                })

            # Training loop
            model.train()
            output_dir = model_path / 'finetuned'
            output_dir.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"Will save model to: {output_dir}")
            logger.info(f"Run artefacts directory: {run_dir}")
            logger.info(f"Using device: {self.device}")
            
            for epoch in range(start_epoch, num_epochs):
                if self._cancel_event.is_set():
                    logger.info("Training cancelled")
                    self._update_run_metadata(run_dir, {
                        'status': 'cancelled',
                        'end_time': datetime.now().isoformat(),
                        'error': 'Training cancelled by user'
                    })
                    self.training_status.update({
                        'is_training': False,
                        'is_cancelling': False,
                        'end_time': datetime.now().isoformat(),
                        'error': 'Training cancelled by user'
                    })
                    return

                # Training phase
                model.train()
                train_loss = 0
                optimizer.zero_grad()  # Clear gradients at start of epoch
                
                # Update current epoch at the start of each epoch (1-based indexing)
                self.training_status.update({
                    'current_epoch': epoch + 1,
                    'total_epochs': num_epochs
                })
                
                num_train_batches = len(train_dataloader)
                for step, batch in enumerate(train_dataloader):
                    if self._cancel_event.is_set():
                        logger.info("Training cancelled")
                        self._update_run_metadata(run_dir, {
                            'status': 'cancelled',
                            'end_time': datetime.now().isoformat(),
                            'error': 'Training cancelled by user'
                        })
                        self.training_status.update({
                            'is_training': False,
                            'is_cancelling': False,
                            'end_time': datetime.now().isoformat(),
                            'error': 'Training cancelled by user'
                        })
                        return

                    try:
                        # Move batch to device and ensure it stays there
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        labels = batch['labels'] if 'labels' in batch else batch['input_ids'].clone()
                        
                        # More frequent CUDA cache clearing
                        if step % 5 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        if not loss.requires_grad:
                            raise RuntimeError("Loss does not require grad; training cannot proceed (model likely frozen or grad disabled).")
                        
                        train_loss += loss.item()
                        loss.backward()
                        
                        if (step + 1) % grad_accum_steps == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                        
                        # Flush remaining gradients at end of epoch
                        if step == num_train_batches - 1 and (step + 1) % grad_accum_steps != 0:
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
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.error(f"OOM Error during training step {step}: {str(e)}")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            raise RuntimeError("GPU Out of Memory. Try reducing batch size or sequence length.") from e
                        raise e
                
                # Validation phase
                model.eval()
                val_loss = 0
                token_correct = 0
                token_total = 0
                pred_texts = []
                ref_texts = []
                with torch.no_grad():
                    for batch in val_dataloader:
                        if self._cancel_event.is_set():
                            logger.info("Training cancelled")
                            self._update_run_metadata(run_dir, {
                                'status': 'cancelled',
                                'end_time': datetime.now().isoformat(),
                                'error': 'Training cancelled by user'
                            })
                            self.training_status.update({
                                'is_training': False,
                                'is_cancelling': False,
                                'end_time': datetime.now().isoformat(),
                                'error': 'Training cancelled by user'
                            })
                            return

                        # Move batch to device and ensure it stays there
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        labels = batch['labels'] if 'labels' in batch else batch['input_ids'].clone()
                        
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=labels
                        )
                        
                        val_loss += outputs.loss.item()
                        if compute_accuracy or compute_bleu or compute_rouge:
                            logits = outputs.logits
                            pred_ids = torch.argmax(logits, dim=-1)
                            label_mask = labels != -100
                            if label_mask.any():
                                if compute_accuracy:
                                    token_correct += (pred_ids[label_mask] == labels[label_mask]).sum().item()
                                    token_total += label_mask.sum().item()
                                if compute_bleu or compute_rouge:
                                    for row_idx in range(labels.size(0)):
                                        row_mask = label_mask[row_idx]
                                        if not row_mask.any():
                                            continue
                                        pred_tokens = pred_ids[row_idx][row_mask].tolist()
                                        label_tokens = labels[row_idx][row_mask].tolist()
                                        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                                        ref_text = tokenizer.decode(label_tokens, skip_special_tokens=True).strip()
                                        if pred_text or ref_text:
                                            pred_texts.append(pred_text)
                                            ref_texts.append(ref_text)

                        current_step += 1
                        progress = (current_step / total_steps) * 100
                        self.training_status['progress'] = round(progress, 2)
                
                val_loss = val_loss / len(val_dataloader)
                epoch_train_loss = train_loss / len(train_dataloader)

                eval_metrics = {}
                if compute_accuracy and token_total > 0:
                    eval_metrics['accuracy'] = token_correct / token_total

                if (compute_bleu or compute_rouge) and pred_texts:
                    try:
                        import evaluate
                        if compute_bleu:
                            bleu_metric = evaluate.load("bleu")
                            bleu_result = bleu_metric.compute(
                                predictions=pred_texts,
                                references=[[ref] for ref in ref_texts]
                            )
                            eval_metrics['bleu'] = bleu_result.get('bleu')
                        if compute_rouge:
                            rouge_metric = evaluate.load("rouge")
                            rouge_result = rouge_metric.compute(
                                predictions=pred_texts,
                                references=ref_texts,
                                use_stemmer=True
                            )
                            for key in ['rouge1', 'rouge2', 'rougeL']:
                                score = rouge_result.get(key)
                                if score is not None:
                                    eval_metrics[key] = score.mid.fmeasure if hasattr(score, 'mid') else score
                    except Exception as e:
                        logger.warning(f"Metric evaluation failed: {e}")

                if eval_hooks:
                    hook_context = {
                        'epoch': epoch + 1,
                        'train_loss': epoch_train_loss,
                        'val_loss': val_loss,
                        'metrics': dict(eval_metrics),
                        'predictions': list(pred_texts),
                        'references': list(ref_texts),
                        'run_id': run_id,
                        'model_id': model_id,
                        'params': params
                    }
                    eval_metrics.update(self._apply_eval_hooks(eval_hooks, hook_context))
                
                # Add metrics to history
                epoch_metrics = {
                    'epoch': epoch + 1,  # Use 1-based epoch numbering
                    'train_loss': epoch_train_loss,
                    'val_loss': val_loss,
                    'run_id': run_id,
                    'timestamp': datetime.now().isoformat(),
                    **eval_metrics
                }
                self.training_status['history'].append(epoch_metrics)
                self._append_metrics(run_dir, epoch_metrics)
                
                # Update status with epoch completion
                self.training_status.update({
                    'current_epoch': epoch + 1,  # Use 1-based epoch numbering
                    'total_epochs': num_epochs,
                    'val_loss': val_loss,
                    'train_loss': epoch_train_loss,
                    'current_metrics': epoch_metrics
                })
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Update best model whenever improvement is found
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"New best validation loss: {val_loss:.4f}. Saving best model...")
                    
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
                            'run_id': run_id,
                            'model_id': model_id,
                            'training_method': training_method,
                            'best_val_loss': best_val_loss,
                            'epochs_completed': epoch + 1,
                            'training_params': params,
                            'save_time': datetime.now().isoformat()
                        }
                        
                        with open(run_dir / 'training_config.json', 'w') as f:
                            json.dump(training_config, f, indent=2)
                            
                        logger.info(f"Saved training configuration to {run_dir / 'training_config.json'}")
                        
                    except Exception as save_error:
                        logger.error(f"Error saving model: {str(save_error)}")

                # Save resume checkpoint if enabled and interval hit
                if checkpoint_enabled and ((epoch + 1) % checkpoint_interval_epochs == 0):
                    checkpoint_state = {
                        'epoch': epoch + 1,
                        'global_step': current_step,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'seed': seed,
                        'rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    }
                    self._save_checkpoint(checkpoint_path, checkpoint_state)
            
            # Save final model state if it's the best one (regardless of checkpoint toggle)
            if val_loss <= best_val_loss:
                logger.info("Final model is the best model. Saving...")
                if training_method in ['lora', 'qlora']:
                    model.save_pretrained(output_dir)
                else:
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
            
            self.training_status.update({
                'is_training': False,
                'is_cancelling': False,
                'progress': 100,
                'end_time': datetime.now().isoformat(),
                'error': None,
                'final_val_loss': val_loss,
                'best_val_loss': best_val_loss
            })

            self._update_run_metadata(run_dir, {
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'final_val_loss': val_loss,
                'best_val_loss': best_val_loss
            })
            
            logger.info(f"Training completed. Model saved at: {output_dir}")
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            try:
                if 'run_dir' in locals():
                    self._update_run_metadata(run_dir, {
                        'status': 'failed',
                        'end_time': datetime.now().isoformat(),
                        'error': str(e)
                    })
            except Exception:
                pass
            self.training_status.update({
                'is_training': False,
                'is_cancelling': False,
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            raise 