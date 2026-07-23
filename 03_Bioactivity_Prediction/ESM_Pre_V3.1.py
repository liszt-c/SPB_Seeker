import pandas as pd
import numpy as np
import argparse
import os
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, matthews_corrcoef,
                            confusion_matrix, balanced_accuracy_score)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Custom dataset class for protein sequences
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=1024):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Compatible with both old and new versions of transformers
        try:
            # New version: direct tokenizer call
            encoding = self.tokenizer(
                sequence,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        except AttributeError:
            # Old version: use encode_plus
            encoding = self.tokenizer.encode_plus(
                sequence,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

# Deep learning model architecture using ESM as backbone
class CPPPredictor(nn.Module):
    def __init__(self, esm_model, hidden_size=256, dropout=0.3):
        super(CPPPredictor, self).__init__()
        self.esm = esm_model
        
        # Freeze ESM parameters
        for param in self.esm.parameters():
            param.requires_grad = False
            
        # Get ESM hidden dimension
        esm_hidden_size = esm_model.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(esm_hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, 1)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the CLS token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

# Training function
def train_model(model_name, hidden_size, dataset_name):
    # Load data
    print(f"Loading data: {dataset_name}...")
    data = pd.read_csv(dataset_name)
    data = data.dropna(subset=['sequence'])
    
    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(dataset_name))[0]
    print(f"Dataset: {base_name} (Positive: {sum(data['label']==1)}, Negative: {sum(data['label']==0)})")
    
    # Initialize ESM tokenizer and model
    print(f"Initializing ESM model: {model_name}, hidden size: {hidden_size}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm_model = AutoModel.from_pretrained(model_name)
    
    # Split dataset
    print("\nSplitting dataset...")
    X = data['sequence'].tolist()
    y = data['label'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create datasets and dataloaders
    train_dataset = ProteinDataset(X_train, y_train, tokenizer, max_length=1024)
    val_dataset = ProteinDataset(X_val, y_val, tokenizer, max_length=1024)
    test_dataset = ProteinDataset(X_test, y_test, tokenizer, max_length=1024)
    
    batch_size = 16  # May need to reduce due to ESM model size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CPPPredictor(esm_model, hidden_size=hidden_size).to(device)
    
    # Loss function and optimizer with class weighting for imbalanced data
    if sum(y_train) > 0:
        pos_weight = torch.tensor([len(y_train) / sum(y_train)]).to(device)
    else:
        pos_weight = torch.tensor([1.0]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training parameters
    num_epochs = 100
    best_val_auc = 0
    patience = 30
    patience_counter = 0
    
    # Training loop
    print("\nStarting model training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            loss = criterion(outputs.flatten(), labels.flatten())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = np.atleast_1d(outputs.squeeze().detach().cpu().numpy())
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        # Compute training metrics
        train_loss /= len(train_loader)
        train_preds = np.array(all_preds) > 0.5
        train_acc = accuracy_score(all_labels, train_preds)
        train_auc = roc_auc_score(all_labels, all_preds)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.flatten(), labels.flatten())
                
                val_loss += loss.item()
                preds = np.atleast_1d(outputs.squeeze().cpu().numpy())
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_preds_binary = np.array(val_preds) > 0.5
        val_acc = accuracy_score(val_labels, val_preds_binary)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            model_dir = f"models/{base_name}_{hidden_size}"
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{model_dir}/best_model.pth")
            print(f"  Saved best model (Val AUC={best_val_auc:.4f}) to {model_dir}")
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"Early stopping triggered! Stopping after {epoch+1} epochs")
            break
    
    # Load best model
    model_dir = f"models/{base_name}_{hidden_size}"
    model.load_state_dict(torch.load(f"{model_dir}/best_model.pth"))
    model.eval()
    
    # Test evaluation
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = np.atleast_1d(outputs.squeeze().cpu().numpy())
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())
    
    test_preds_binary = np.array(test_preds) > 0.5
    test_acc = accuracy_score(test_labels, test_preds_binary)
    test_precision = precision_score(test_labels, test_preds_binary, zero_division=0)
    test_recall = recall_score(test_labels, test_preds_binary, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds_binary, zero_division=0)
    test_auc = roc_auc_score(test_labels, test_preds)
    test_mcc = matthews_corrcoef(test_labels, test_preds_binary)
    
    # Save results
    results = {
        'Accuracy': test_acc,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1 Score': test_f1,
        'AUC': test_auc,
        'MCC': test_mcc,
        'Confusion Matrix': confusion_matrix(test_labels, test_preds_binary)
    }
    
    results_dir = f"results/{base_name}_{hidden_size}"
    os.makedirs(results_dir, exist_ok=True)
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"{results_dir}/performance.csv", index=False)
    
    print(f"\nTest set performance (hidden_size={hidden_size}):")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  MCC: {test_mcc:.4f}")
    
    # Save tokenizer and model
    tokenizer.save_pretrained(f"{model_dir}/tokenizer")
    torch.save(model.state_dict(), f"{model_dir}/final_model.pth")
    
    # Save model configuration
    with open(f"{model_dir}/esm_config.pth", 'w') as f:
        f.write(f"hidden_size: {hidden_size}\n")
        f.write(f"base_model: {model_name}\n")
        f.write(f"dataset: {base_name}\n")
    
    print(f"\nTraining complete! Model saved to {model_dir}/")

# Prediction function
def predict_data(input_csv, model_dir, output_csv=None):
    if not output_csv:
        input_base = os.path.splitext(os.path.basename(input_csv))[0]
        model_base = os.path.basename(model_dir)
        output_csv = f"{input_base}_{model_base}_predictions.csv"
    
    # Load model configuration
    config_path = f"{model_dir}/esm_config.pth"
    if not os.path.exists(config_path):
        print(f"Error: Model configuration file {config_path} not found")
        return
    
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(': ')
                config[key] = value
    
    hidden_size = int(config.get('hidden_size', 4))
    base_model = config.get('base_model', './facebook/esm2_t33_650M_UR50D')
    
    # Load tokenizer
    tokenizer_path = f"{model_dir}/tokenizer"
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer path {tokenizer_path} not found")
        return
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error: Cannot load tokenizer: {e}")
        return
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        esm_model = AutoModel.from_pretrained(base_model)
    except Exception as e:
        print(f"Error: Cannot load ESM model {base_model}: {e}")
        return
    
    model = CPPPredictor(esm_model, hidden_size=hidden_size).to(device)
    
    # Load weights
    model_path = f"{model_dir}/final_model.pth"
    if not os.path.exists(model_path):
        model_path = f"{model_dir}/best_model.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error: Cannot load model weights: {e}")
        return
    
    # Load data
    print(f"Reading prediction data: {input_csv}")
    new_data = pd.read_csv(input_csv)
    
    if 'sequence' not in new_data.columns:
        print("Error: Input file must contain a 'sequence' column")
        return
        
    sequences_list = new_data['sequence'].tolist()
    print(f"Samples to predict: {len(sequences_list)}")
    
    # Create dataset
    dataset = ProteinDataset(sequences_list, [0]*len(sequences_list), tokenizer, max_length=1024)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Predict
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = np.atleast_1d(outputs.squeeze().cpu().numpy())
            predictions.extend(preds)
    
    # Add predictions to dataframe
    new_data['prediction'] = predictions
    new_data['prediction_binary'] = (np.array(predictions) > 0.5).astype(int)
    
    # Save results
    new_data.to_csv(output_csv, index=False)
    print(f"Prediction results saved to {output_csv}")
    
    return output_csv

# Main function
def main():
    parser = argparse.ArgumentParser(description='ESM-based CPP prediction tool with ablation study support')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ESM-based deep learning model')
    train_parser.add_argument('--data', default='ToxiDataset_Length_Filtered.csv', 
                            help='Input CSV file with sequence column (default: ToxiDataset_Length_Filtered.csv)')
    train_parser.add_argument('--model', default='./facebook/esm2_t33_650M_UR50D', 
                            help='ESM model path (default: ./facebook/esm2_t33_650M_UR50D)')
    train_parser.add_argument('--hidden_size', type=int, default=1024, 
                            help='Hidden layer size for MLP (default: 1024)')
    
    # Predict command
    predict_parser = subparsers.add_parser('pre', help='Predict using trained model')
    predict_parser.add_argument('--input', required=True, help='Input CSV file with sequence column')
    predict_parser.add_argument('--model_dir', required=True, help='Directory containing trained model')
    predict_parser.add_argument('--output', help='Output CSV file', default=None)
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print(f"Starting model training...")
        print(f"  Dataset: {args.data}")
        print(f"  ESM model: {args.model}")
        print(f"  Hidden size: {args.hidden_size}")
        train_model(args.model, args.hidden_size, args.data)
    elif args.command == 'pre':
        print("Starting prediction...")
        predict_data(args.input, args.model_dir, args.output)
    else:
        print("Please specify a command: 'train' or 'pre'")
        print("\nExamples:")
        print("  Train: python ESM_Pre_V3.1.py train --data ToxiDataset_Length_Filtered.csv --hidden_size 1024")
        print("  Predict: python ESM_Pre_V3.1.py pre --input test.csv --model_dir models/ToxiDataset_Length_Filtered_1024")

if __name__ == "__main__":
    main()