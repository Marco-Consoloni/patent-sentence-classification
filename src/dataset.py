import torch
import pandas as pd
from torch.utils.data import Dataset

class PatentDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.data = pd.read_excel(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      sent = self.data.iloc[idx]['sent']
      inputs = self.tokenizer(
          sent,
          padding='max_length',
          max_length=self.max_length,
          truncation=True,
          return_tensors='pt'
          )

      return {
          'input_ids': inputs['input_ids'].squeeze(0),  # Remove batch dimension
          'attention_mask': inputs['attention_mask'].squeeze(0),
          'token_type_ids': inputs['token_type_ids'].squeeze(0),
          'labels': torch.tensor(self.data.iloc[idx]['sent_class'], dtype=torch.long)
          }