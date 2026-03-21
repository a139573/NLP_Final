import torch.nn as nn
from transformers import AutoModel, AutoConfig

class UniversalTransformerClassifier(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', n_classes=20, dropout_rate=0.3):
        super().__init__()
        # AutoModel dynamically loads the correct architecture 
        # (works for BERT, DistilBERT, RoBERTa, FinBERT, etc.)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # AutoConfig dynamically fetches the hidden size (usually 768 for base models)
        config = AutoConfig.from_pretrained(model_name)
        self.fc = nn.Linear(config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the representation of the first token. 
        # This handles [CLS] for BERT/DistilBERT/FinBERT and <s> for RoBERTa automatically.
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        return self.fc(self.dropout(cls_output))