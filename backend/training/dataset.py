from torch.utils.data import Dataset
import torch
from config.settings import MAX_LENGTH
import logging

logger = logging.getLogger('training')

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, input_field='input', output_field='output'):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = MAX_LENGTH
        self.input_field = input_field
        self.output_field = output_field
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
                    # Check if this is messages format
                    if self.input_field == 'messages' and 'messages' in item:
                        input_text, output_text = self._extract_from_messages(item['messages'])
                    else:
                        # Standard JSON format with configurable fields
                        input_text = str(item.get(self.input_field, ''))
                        output_text = str(item.get(self.output_field, ''))
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

    def _extract_from_messages(self, messages):
        """Extract input and output from OpenAI-style messages array"""
        if not isinstance(messages, list) or not messages:
            return '', ''
        
        system_parts = []
        user_content = ''
        assistant_content = ''
        
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                if content:
                    system_parts.append(str(content))
            elif role == 'user':
                user_content = content
            elif role == 'assistant':
                assistant_content = content
        
        combined_input = str(user_content)
        if system_parts:
            combined_input = "\n\n".join([*system_parts, combined_input]).strip()

        return combined_input, str(assistant_content)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_text = f"{item['input']}{self.tokenizer.sep_token}"
        output_text = str(item['output'])
        
        try:
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                if getattr(self.tokenizer, "eos_token", None) is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=False
            )["input_ids"]
            output_ids = self.tokenizer(
                output_text,
                add_special_tokens=False,
                truncation=False
            )["input_ids"]

            if getattr(self.tokenizer, "eos_token_id", None) is not None:
                output_ids = output_ids + [self.tokenizer.eos_token_id]

            if not output_ids:
                output_ids = [self.tokenizer.eos_token_id or 0]

            available_for_prompt = max(1, self.max_length - len(output_ids))
            if len(prompt_ids) > available_for_prompt:
                prompt_ids = prompt_ids[:available_for_prompt]

            available_for_output = max(1, self.max_length - len(prompt_ids))
            if len(output_ids) > available_for_output:
                output_ids = output_ids[:available_for_output]

            input_ids = prompt_ids + output_ids
            attention_mask = [1] * len(input_ids)

            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            if len(input_ids) < self.max_length:
                pad_len = self.max_length - len(input_ids)
                input_ids = input_ids + [pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

            labels = [-100] * self.max_length
            prompt_len = min(len(prompt_ids), self.max_length)
            for i in range(prompt_len, min(prompt_len + len(output_ids), self.max_length)):
                labels[i] = input_ids[i]

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        except Exception as e:
            logger.error(f"Error encoding item {idx}: {str(e)}")
            # Return a dummy item in case of error to avoid training failure
            # This is a fallback - the item will be mostly masked
            dummy = torch.zeros(self.max_length, dtype=torch.long)
            dummy[0] = self.tokenizer.bos_token_id or 0  # Start token
            return {
                'input_ids': dummy,
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.full((self.max_length,), -100, dtype=torch.long)
            } 