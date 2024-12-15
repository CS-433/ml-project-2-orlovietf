import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import numpy as np

# Load the predictions
val_preds = pd.read_csv('/home/ml4science0/novozymes/predictions/esm2_t6_8M_UR50D_val_preds.csv')

# load the model from pth_models folder and get its architecture withouth the class ProteinModel
model = torch.load('/home/ml4science0/novozymes/pth_models/esm2_t6_8M_UR50D_model.pth')

# Get the layers of the model and print them
print(model)

# Assuming the CSV has columns 'true' and 'pred'
true_values = val_preds['tm']
pred_values = val_preds['preds']

# Calculate performance metrics
pcc, _ = pearsonr(true_values, pred_values)
scc, _ = spearmanr(true_values, pred_values)
rmse = np.sqrt(mean_squared_error(true_values, pred_values))
mae = mean_absolute_error(true_values, pred_values)

# Store the metrics
metrics = {
    'PCC': pcc,
    'SCC': scc,
    'RMSE': rmse,
    'MAE': mae
}

# Save metrics to a file
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('/home/ml4science0/novozymes/results/esm2_t6_8M_UR50D_metrics.csv', index=False)

# Print the metrics
print(metrics_df)