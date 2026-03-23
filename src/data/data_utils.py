import torch
from collections import Counter
from torch.utils.data import Dataset

class Vocabulary:
    def __init__(self, texts, max_size=10000):
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.lower().split())
        
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in word_counts.most_common(max_size):
            self.vocab[word] = len(self.vocab)
            
    def text_to_indices(self, text, max_len=128):
        tokens = text.lower().split()
        indices = [self.vocab.get(t, 1) for t in tokens[:max_len]]
        indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

class LSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab_handler, max_len=128):
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else texts
        self.labels = labels
        self.vh = vocab_handler
        self.max_len = max_len
        
    def __len__(self): 
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Passing max_len to text_to_indices to ensure consistent padding
        return self.vh.text_to_indices(self.texts[idx], self.max_len), torch.tensor(self.labels[idx], dtype=torch.float)

class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self): 
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Convert text to string just in case there are float/NaN values
        enc = self.tokenizer(
            str(self.texts[idx]), 
            truncation=True,
            padding='max_length', 
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        # squeeze(0) removes the batch dimension added by return_tensors='pt'
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }