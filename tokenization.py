import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
import torch


model_checkpoint = "facebook/esm2_t12_35M_UR50D"

df = pd.read_csv("train.csv")
sequences = df["protein_sequence"].tolist()
tm = df["tm"].values

print("Number of sequences:", len(sequences))

print("First few sequences:")
for seq in sequences[:5]:
    print(seq)

train_sequences, val_sequences, train_tm, val_tm = train_test_split(sequences, tm, test_size=0.2, shuffle=True, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_tokenized = tokenizer(train_sequences, padding=True, return_tensors="pt")
val_tokenized = tokenizer(val_sequences, padding=True, return_tensors="pt")

print("Tokenization finished")

torch.save(train_tokenized, "tokenized_seqs/train_tokenized.pt")
torch.save(val_tokenized, "tokenized_seqs/val_tokenized.pt")

print("Tokenized sequences saved")

train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(val_tokenized)

train_dataset = train_dataset.add_column("melt_temp", train_tm)
test_dataset = test_dataset.add_column("melt_temp", val_tm)

train_dataset.save_to_disk("tokenized_seqs/train_dataset")
test_dataset.save_to_disk("tokenized_seqs/test_dataset")