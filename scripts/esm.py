from transformers import AutoModel
import torch.nn as nn


class ProteinModel(nn.Module):
    def __init__(self, model_checkpoint):
        super(ProteinModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_checkpoint)
        self.fc1 = nn.Linear(320, 120)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        out = self.fc1(cls_token)
        out = self.fc2(out)
        return out
    