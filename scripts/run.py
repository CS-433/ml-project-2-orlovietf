from baseline_model import ProteinModel
from datasets import ProteinDataset
from train import *
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModel
import torch.nn as nn
from scipy.stats import spearmanr

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("/home/ml4science0/novozymes/train_updated.csv")
    test_df = pd.read_csv("/home/ml4science0/novozymes/test.csv")

    sequences = df["protein_sequence"].tolist()
    tm = df["tm"].values

    test_sequences = test_df["protein_sequence"].tolist()

    train_sequences, val_sequences, train_tm, val_tm = train_test_split(sequences, tm, test_size=0.2, shuffle=True, random_state=42)

    model_checkpoint = "facebook/esm2_t6_8M_UR50D"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    train_tokenized = tokenizer(train_sequences, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    val_tokenized = tokenizer(val_sequences, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    test_tokenized = tokenizer(test_sequences, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
            
    train_dataset = ProteinDataset(train_tokenized, train_tm)
    val_dataset = ProteinDataset(val_tokenized, val_tm)
    test_dataset = ProteinDataset(test_tokenized)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = ProteinModel(model_checkpoint).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

    tms, preds = inference(model, val_loader, device)

    torch.save(model, "/home/ml4science0/novozymes/pth_models/esm2_t6_8M_UR50D_model.pth")
    pd.DataFrame({"tm": tms, "preds": preds}).to_csv("/home/ml4science0/novozymes/predictions/esm2_t6_8M_UR50D_val_preds.csv", index=False)



if __name__ == "__main__":
    main()