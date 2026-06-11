import json

def create_notebook():
    cells = []
    
    def add_markdown(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })
        
    def add_code(code):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.split("\n")]
        })

    add_markdown("# Multimodal Dropout Prediction Model\nHighly Optimized: Mixed Precision, Checkpointing, and Vectorized Pipeline.")
    
    add_code("""!pip install -q transformers scikit-learn pandas numpy torch tqdm""")
    
    add_code("""import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import os
import torch.utils.checkpoint as checkpoint

# Ensure deterministic behavior
torch.manual_seed(42)
np.random.seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")""")

    add_markdown("## 1. Data Loading")
    add_code("""DATA_DIR = '/kaggle/input/retina-ai-hackathon' if os.path.exists('/kaggle/input/retina-ai-hackathon') else '.'

train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
test_df = pd.read_csv(f'{DATA_DIR}/test.csv')
attendance_df = pd.read_csv(f'{DATA_DIR}/Attendance_series.csv')
notes_df = pd.read_csv(f'{DATA_DIR}/Counsellor_notes.csv')""")

    add_markdown("## 2. Preprocessing\n### Tabular Preprocessing (No Leakage)")
    add_code("""def preprocess_tabular(train, test):
    train_proc = train.copy()
    test_proc = test.copy()
    
    num_cols = ['commute_time_mins', 'screen_time_hours'] + [c for c in train.columns if 'cgpa' in c or 'backlog' in c]
    cat_cols = ['branch', 'gender', 'hostel_status', 'family_income', 'parent_education']
    
    # Impute missing
    num_imputer = SimpleImputer(strategy='median')
    train_proc[num_cols] = num_imputer.fit_transform(train_proc[num_cols])
    test_proc[num_cols] = num_imputer.transform(test_proc[num_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    train_proc[cat_cols] = cat_imputer.fit_transform(train_proc[cat_cols])
    test_proc[cat_cols] = cat_imputer.transform(test_proc[cat_cols])
    
    # Categorical Encoding (Fit on train, map to test)
    for col in cat_cols:
        le = LabelEncoder()
        train_proc[col] = le.fit_transform(train_proc[col])
        # Handle unseen categories safely
        test_proc[col] = test_proc[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')
        test_proc[col] = le.transform(test_proc[col])
        
    # Scale
    scaler = StandardScaler()
    train_proc[num_cols] = scaler.fit_transform(train_proc[num_cols])
    test_proc[num_cols] = scaler.transform(test_proc[num_cols])
    
    feature_cols = num_cols + cat_cols + ['scholarship', 'part_time_job']
    return train_proc, test_proc, feature_cols

train_tab, test_tab, tab_features = preprocess_tabular(train_df, test_df)""")

    add_markdown("### Time-Series Preprocessing (Pre-allocated Numpy Array)")
    add_code("""def preprocess_attendance(attendance, students):
    att_sorted = attendance.sort_values(by=['student_id', 'semester', 'week'])
    weekly_att = att_sorted.groupby(['student_id', 'semester', 'week'])['attendance_pct'].mean().reset_index()
    
    max_len = 32
    # Pre-allocate array for speed
    student_idx_map = {stu: i for i, stu in enumerate(students)}
    seq_array = np.zeros((len(students), max_len))
    
    grouped = weekly_att.groupby('student_id')
    for stu, group in grouped:
        if stu in student_idx_map:
            vals = group['attendance_pct'].values
            idx = student_idx_map[stu]
            seq_array[idx, :min(len(vals), max_len)] = vals[:max_len]
            
    # Return a dict mapping for easy lookup
    return {stu: seq_array[i] for i, stu in enumerate(students)}

all_students = pd.concat([train_df['student_id'], test_df['student_id']]).unique()
attendance_sequences = preprocess_attendance(attendance_df, all_students)""")

    add_markdown("### Text Preprocessing (Dictionary Mapping)")
    add_code("""# Aggregate notes per student
notes_grouped = notes_df.groupby('student_id')['counsellor_note'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
notes_dict = dict(zip(notes_grouped['student_id'], notes_grouped['counsellor_note']))

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')""")

    add_markdown("## 3. Dataset & DataLoader (Optimized Pre-Tokenization)")
    add_code("""class MultimodalDataset(Dataset):
    def __init__(self, tab_df, tab_features, att_seqs, student_ids, labels=None, max_len=128):
        self.tabular = torch.tensor(tab_df[tab_features].values, dtype=torch.float32)
        self.attendance = torch.tensor([att_seqs[stu] for stu in student_ids], dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long) if labels is not None else None
        
        # Pre-tokenize all texts during init to save batch loop memory/CPU
        texts = [notes_dict.get(stu, "no notes") for stu in student_ids]
        self.tokenized_texts = tokenizer(texts, padding='max_length', truncation=True, 
                                       max_length=max_len, return_tensors='pt')
        
    def __len__(self):
        return len(self.tabular)
    
    def __getitem__(self, idx):
        item = {
            'tabular': self.tabular[idx],
            'attendance': self.attendance[idx],
            'input_ids': self.tokenized_texts['input_ids'][idx],
            'attention_mask': self.tokenized_texts['attention_mask'][idx]
        }
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item

from sklearn.model_selection import train_test_split
train_split, val_split = train_test_split(train_tab, test_size=0.2, random_state=42, stratify=train_tab['dropout_risk'])

train_dataset = MultimodalDataset(train_split, tab_features, attendance_sequences, train_split['student_id'].values, train_split['dropout_risk'])
val_dataset = MultimodalDataset(val_split, tab_features, attendance_sequences, val_split['student_id'].values, val_split['dropout_risk'])
test_dataset = MultimodalDataset(test_tab, tab_features, attendance_sequences, test_tab['student_id'].values)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Larger batch due to optimizations
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)""")

    add_markdown("## 4. Model Architecture (BatchNorm, Gradient Checkpointing, Balanced Fusion)")
    add_code("""class MultimodalDropoutPredictor(nn.Module):
    def __init__(self, tabular_dim, num_classes=3):
        super().__init__()
        # Tabular (64 dim output)
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Attendance (Project to 64 dim)
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        
        # Text (Project to 64 dim)
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.distilbert.gradient_checkpointing_enable() # Memory optimization
        
        for param in self.distilbert.parameters():
            param.requires_grad = False
        for param in self.distilbert.transformer.layer[-2:].parameters(): # Unfreeze top 2
            param.requires_grad = True
            
        self.text_fc = nn.Linear(768, 64)
        
        # Fusion Layer (64 + 64 + 64 = 192)
        self.fusion = nn.Sequential(
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, tabular, attendance, input_ids, attention_mask):
        tab_emb = self.tab_mlp(tabular)
        
        att_unsqueezed = attendance.unsqueeze(-1)
        _, (hn, _) = self.lstm(att_unsqueezed)
        att_emb = hn[-1]
        
        bert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = bert_output.last_hidden_state[:, 0, :]
        text_emb = torch.relu(self.text_fc(cls_token))
        
        fused = torch.cat((tab_emb, att_emb, text_emb), dim=1)
        logits = self.fusion(fused)
        return logits

model = MultimodalDropoutPredictor(tabular_dim=len(tab_features)).to(device)""")

    add_markdown("## 5. Training Loop (Mixed Precision)")
    add_code("""criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
scaler = GradScaler()

epochs = 5
best_f1 = 0.0

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        optimizer.zero_grad()
        tab, att = batch['tabular'].to(device), batch['attendance'].to(device)
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        with autocast():
            logits = model(tab, att, ids, mask)
            loss = criterion(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            tab, att = batch['tabular'].to(device), batch['attendance'].to(device)
            ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with autocast():
                logits = model(tab, att, ids, mask)
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Macro F1: {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pth')
        print("-> Saved best model!")""")

    add_markdown("## 6. Inference & Submission")
    add_code("""model.load_state_dict(torch.load('best_model.pth'))
model.eval()

test_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inference"):
        tab, att = batch['tabular'].to(device), batch['attendance'].to(device)
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        
        with autocast():
            logits = model(tab, att, ids, mask)
        preds = torch.argmax(logits, dim=1)
        test_preds.extend(preds.cpu().numpy())
        
submission = pd.DataFrame({'student_id': test_tab['student_id'], 'dropout_risk': test_preds})
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('kaggle_solution.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)

if __name__ == "__main__":
    create_notebook()
