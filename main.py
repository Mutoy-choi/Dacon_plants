import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Assuming iTransformer is properly installed in your environment
from iTransformer.iTransformer import iTransformer

# CUDA availability check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the datasets
train_df = pd.read_csv('./processed_train.csv')
test_df = pd.read_csv('./processed_test.csv')

# Data preprocessing
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
train_df['day_of_year'] = train_df['timestamp'].dt.dayofyear
train_df['year'] = train_df['timestamp'].dt.year

# Features for prediction
features = ['day_of_year', 'year', 'supply(kg)', 'item', 'corporation', 'location', 'is_holiday']
# Encode categorical features
label_encoders = {}
for feature in ['item', 'corporation', 'location']:
    le = LabelEncoder()
    train_df[feature] = le.fit_transform(train_df[feature])
    test_df[feature] = le.transform(test_df[feature])
    label_encoders[feature] = le

X = train_df[features].values
y = train_df['price(원/kg)'].values

sequence_length = 96
# Split data into sequences
X_sequences = np.array([X[i:i+sequence_length] for i in range(len(X)-sequence_length+1)])
y_sequences = np.array([y[i:i+sequence_length] for i in range(len(y)-sequence_length+1)])

# Split into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# Prepare the supply feature and label
features_supply = ['day_of_year', 'year', 'item', 'corporation', 'location', 'is_holiday']
y_supply = train_df['supply(kg)'].values
X_supply = train_df[features_supply].values

# Split supply data into sequences
X_supply_sequences = np.array([X_supply[i:i+sequence_length] for i in range(len(X_supply)-sequence_length+1)])
y_supply_sequences = np.array([y_supply[i:i+sequence_length] for i in range(len(y_supply)-sequence_length+1)])

# Split into training and validation sets for supply
X_supply_train, X_supply_valid, y_supply_train, y_supply_valid = train_test_split(
    X_supply_sequences, y_supply_sequences, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and move to GPU if available
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)

# Initialize the iTransformer model
model = iTransformer(
    num_variates=len(features),
    lookback_len=sequence_length,
    dim=64,
    depth=4,
    heads=4,
    dim_head=32,
    pred_length=sequence_length,
    num_tokens_per_variate=1,
    use_reversible_instance_norm=True
)
model = model.to(device)

# Training configurations
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_valid_tensor)
        val_loss = criterion(val_outputs, y_valid_tensor)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

# Processing for test set predictions
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
test_df['day_of_year'] = test_df['timestamp'].dt.dayofyear
test_df['year'] = test_df['timestamp'].dt.year

X_test = test_df[features].values
X_test_sequences = np.array([X_test[i:i+sequence_length] for i in range(len(X_test)-sequence_length+1)])
X_test_tensor = torch.tensor(X_test_sequences, dtype=torch.float32).to(device)

# Predict on the test set
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)

# Reshape predictions to match submission format
test_preds = test_preds.cpu().numpy().flatten()

# Prepare submission file
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['price(원/kg)'] = test_preds[:len(sample_submission)]  # Ensure the length matches the sample submission

# Save the submission file
sample_submission.to_csv('submission.csv', index=False)