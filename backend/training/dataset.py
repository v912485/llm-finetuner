from torch.utils.data import Dataset
import torch
from config.settings import MAX_LENGTH
import logging

logger = logging.getLogger('training')

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = MAX_LENGTH
        self.stats = {
            'total_items': len(data),
            'valid_items': 0,
            'invalid_items': 0,
            'empty_inputs': 0,
            'empty_outputs': 0,
            'truncated_items': 0
        }

        # Check if tokenizer has a separator token, use default if not
        if not hasattr(self.tokenizer, 'sep_token') or self.tokenizer.sep_token is None:
            self.tokenizer.sep_token = self.tokenizer.eos_token or '\n'
            logger.warning(f"Tokenizer has no separator token, using '{self.tokenizer.sep_token}' instead")

        # Process each item based on its format
        for item in data:
            try:
                if isinstance(item, dict):
                    # JSON format
                    input_text = str(item.get('input', ''))
                    output_text = str(item.get('output', ''))
                elif isinstance(item, list) and len(item) >= 2:
                    # CSV format
                    input_text = str(item[0])
                    output_text = str(item[1])
                elif isinstance(item, str):
                    # TXT format - split on common delimiters
                    parts = item.split(':')
                    if len(parts) > 1:
                        input_text = parts[0].strip()
                        output_text = ':'.join(parts[1:]).strip()
                    else:
                        input_text = item.strip()
                        output_text = ''
                else:
                    logger.warning(f"Skipping item with unsupported format: {type(item)}")
                    self.stats['invalid_items'] += 1
                    continue

                # Validate input and output
                if not input_text:
                    self.stats['empty_inputs'] += 1
                    if not output_text:
                        self.stats['invalid_items'] += 1
                        continue
                
                if not output_text:
                    self.stats['empty_outputs'] += 1

                # Check if the combined text would be truncated
                combined_text = f"{input_text}{self.tokenizer.sep_token}{output_text}"
                encoding = self.tokenizer(combined_text, truncation=False, add_special_tokens=True)
                if len(encoding['input_ids']) > self.max_length:
                    self.stats['truncated_items'] += 1
                    logger.debug(f"Item will be truncated: {len(encoding['input_ids'])} > {self.max_length}")

                # Add to dataset
                self.data.append({
                    'input': input_text,
                    'output': output_text
                })
                self.stats['valid_items'] += 1
                
            except Exception as e:
                logger.error(f"Error processing dataset item: {str(e)}")
                self.stats['invalid_items'] += 1
        
        # Log dataset statistics
        logger.info(f"Dataset loaded with {self.stats['valid_items']} valid items")
        logger.info(f"Dataset stats: {self.stats}")
        
        if self.stats['valid_items'] == 0:
            logger.error("No valid items found in dataset!")
            raise ValueError("Dataset contains no valid items")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"{item['input']}{self.tokenizer.sep_token}{item['output']}"
        
        try:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            }
        except Exception as e:
            logger.error(f"Error encoding item {idx}: {str(e)}")
            # Return a dummy item in case of error to avoid training failure
            # This is a fallback - the item will be mostly masked
            dummy = torch.zeros(self.max_length, dtype=torch.long)
            dummy[0] = self.tokenizer.bos_token_id or 0  # Start token
            return {
                'input_ids': dummy,
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
            } 