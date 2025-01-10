from torch.utils.data import Dataset
import torch
from config.settings import MAX_LENGTH

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = MAX_LENGTH

        # Process each item based on its format
        for item in data:
            if isinstance(item, dict):
                # JSON format
                self.data.append({
                    'input': str(item.get('input', '')),
                    'output': str(item.get('output', ''))
                })
            elif isinstance(item, list):
                # CSV format
                self.data.append({
                    'input': str(item[0]),
                    'output': str(item[1])
                })
            elif isinstance(item, str):
                # TXT format - split on common delimiters
                parts = item.split(':')
                if len(parts) > 1:
                    self.data.append({
                        'input': parts[0].strip(),
                        'output': ':'.join(parts[1:]).strip()
                    })
                else:
                    self.data.append({
                        'input': item.strip(),
                        'output': ''
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"{item['input']}{self.tokenizer.sep_token}{item['output']}"
        
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