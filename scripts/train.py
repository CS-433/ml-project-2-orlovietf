import torch

import torch.nn as nn
import torch.optim as optim

from scipy.stats import spearmanr

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tm = batch["tm"].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(1), tm)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        model.eval()
        val_loss = 0.0

        total_preds = []
        total_tms = []
        with torch.no_grad():
            total_loss = 0
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                tm = batch["tm"].float().to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(1), tm)
                total_loss += loss.item()
                val_loss += loss.item()

                total_preds += outputs.squeeze(1).cpu().numpy().tolist()
                total_tms += tm.cpu().numpy().tolist()
        
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')

        spearman_corr = spearmanr(total_preds, total_tms)
        print(f'Spearman Correlation: {spearman_corr.correlation:.4f}')
    
    return model


def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    tms = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids, attention_mask)
            predictions += outputs.squeeze(1).cpu().numpy().tolist()
            if "tm" in batch:
                tm = batch["tm"].float().to(device)
                tms += tm.cpu().numpy().tolist()
    
    return tms, predictions