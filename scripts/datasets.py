from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, sequences, tm=None):
        self.input_ids = sequences["input_ids"]
        self.attention_mask = sequences["attention_mask"]
        self.tm = tm

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.tm is None:
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx]
            }
        else:
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "tm": self.tm[idx]
            }