import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from collections import defaultdict
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("[INFO] Tokenizer loaded successfully.")

# ======================== DATASET ======================== #
class StanceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, subset_size=None):
        print("[INFO] Initializing dataset...")
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = "Regrading'{s1}' the opinion '{s2}' most likely expresses a  [MASK] stance"
        
        self.label_map = {"AGAINST": 1, "FAVOR": 2, "NEUTRAL": 0}
        
        self.data = self.data[['Target', 'Tweet', 'Stance']].dropna()
        self.data['Stance'] = self.data['Stance'].map(self.label_map)
        self.data = self.data.dropna(subset=['Stance']).astype({'Stance': int})
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

        if subset_size and subset_size < len(self.data):
            self.data = self.data.head(subset_size)

        print(f"[INFO] Dataset initialized with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        target, comment, stance = row['Target'], row['Tweet'], row['Stance']

        s1_encoding = self.tokenizer(target, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        s2_encoding = self.tokenizer(comment, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        prompt_text = f"The target '{target}' is described as '{comment}'. This indicates a stance of: [MASK]"
        p_encoding = self.tokenizer(prompt_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return s1_encoding["input_ids"].squeeze(), s1_encoding["attention_mask"].squeeze(), \
               s2_encoding["input_ids"].squeeze(), s2_encoding["attention_mask"].squeeze(), \
               torch.tensor(stance, dtype=torch.long), \
               p_encoding["input_ids"].squeeze(), p_encoding["attention_mask"].squeeze(), \
               target

train_dataset = StanceDataset("C:\\Users\\HP\\Desktop\\ezsd\\final_train_dataset.csv", tokenizer, subset_size=1200)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
print(f"[INFO] DataLoader ready. Batch size: 4")

# ======================== MODEL ======================== #
class GMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=3072):
        super(GMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim // 2)
        self.spatial_proj = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, input_dim)
    
    def forward(self, x):
        z = self.fc1(x)
        z = self.activation(z)
        u, v = torch.chunk(z, chunks=2, dim=-1)
        v = self.spatial_proj(v)
        gated_output = u * v
        gated_output = self.norm(gated_output)
        y = self.fc2(gated_output)
        return y

class EZSDCPModel(nn.Module):
    def __init__(self, hidden_dim=768):
        super(EZSDCPModel, self).__init__()
        print("[INFO] Initializing EZSD-CP Model...")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.gmlp = GMLP(hidden_dim)
        self.channel_gate = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 3, 3)
        print("[INFO] Model initialized successfully.")

    def forward(self, s1_input_ids, s1_attention_mask, p_input_ids, p_attention_mask, s2_input_ids, s2_attention_mask):
        E1 = self.bert(s1_input_ids, attention_mask=s1_attention_mask).last_hidden_state[:, 0, :]
        Ep = self.bert(p_input_ids, attention_mask=p_attention_mask).last_hidden_state[:, 0, :]
        W = self.gmlp(Ep)
        Ep_channel = Ep * self.channel_gate(Ep).sigmoid()
        E_prime_p = W * Ep_channel
        E2 = self.bert(s2_input_ids, attention_mask=s2_attention_mask).last_hidden_state[:, 0, :]
        combined = torch.cat((E_prime_p, E1, E2), dim=1)
        logits = self.fc(combined)
        return logits, combined

model = EZSDCPModel().to(device)

# ======================== TRAINING ======================== #
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
λc = 1.0
λn = 0.1

def stance_contrastive_loss(embeddings, labels, temperature=0.1):
    cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    numerator = torch.exp(cosine_sim / temperature) * positive_mask
    denominator = torch.exp(cosine_sim / temperature).sum(dim=1, keepdim=True)
    loss = -torch.log(numerator.sum(dim=1) / denominator.squeeze())
    return loss.mean()

print("[INFO] Starting training...")
f1_scores_by_target = defaultdict(list)
overall_f1_scores = []

for epoch in range(1):
    print(f"\n[INFO] Epoch {epoch+1} started.")
    model.train()
    all_preds, all_labels, all_targets = [], [], []
    for i, batch in enumerate(train_loader):
        print(f"[INFO] Processing batch {i+1}/{len(train_loader)}...")
        s1_input_ids, s1_attention_mask, s2_input_ids, s2_attention_mask, labels, p_input_ids, p_attention_mask, targets = batch
        s1_input_ids, s1_attention_mask = s1_input_ids.to(device), s1_attention_mask.to(device)
        s2_input_ids, s2_attention_mask = s2_input_ids.to(device), s2_attention_mask.to(device)
        p_input_ids, p_attention_mask = p_input_ids.to(device), p_attention_mask.to(device)
        labels = labels.to(device)

        logits, embeddings = model(s1_input_ids, s1_attention_mask, p_input_ids, p_attention_mask, s2_input_ids, s2_attention_mask)
        loss_ce = criterion(logits, labels)
        loss_contrast = stance_contrastive_loss(embeddings, labels) if len(embeddings) > 1 else torch.tensor(0.0, device=device)
        loss = λc * loss_ce + λn * loss_contrast

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_targets.extend(targets)

        print(f"[INFO] Batch {i+1} - CE Loss: {loss_ce.item():.4f}, Contrastive Loss: {loss_contrast.item():.4f}, Total Loss: {loss.item():.4f}")

    overall_f1 = f1_score(all_labels, all_preds, average='macro')
    overall_f1_scores.append(overall_f1)

    df = pd.DataFrame({'target': all_targets, 'label': all_labels, 'pred': all_preds})
    print("[INFO] Calculating per-target F1 scores...")
    for tgt in df['target'].unique():
        f1 = f1_score(df[df['target'] == tgt]['label'], df[df['target'] == tgt]['pred'], average='macro')
        f1_scores_by_target[tgt].append(f1)

    print(f"[INFO] Epoch {epoch+1} completed.")
    print(f"[RESULT] Overall F1: {overall_f1:.4f}")
    for tgt in f1_scores_by_target:
        print(f"[RESULT] {tgt}: F1 = {f1_scores_by_target[tgt][-1]:.4f}")

# ======================== PREDICT FUNCTION ======================== #
def predict_stance(model, tokenizer, target, comment, device="cpu"):
    model.eval()
    print(f"[INFO] Predicting stance for: Target = '{target}', Comment = '{comment}'")
    with torch.no_grad():
        s1 = tokenizer(target, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
        s2 = tokenizer(comment, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
        p = tokenizer(f"The target '{target}' is described as '{comment}'. This indicates a stance of: [MASK]", return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)

        logits, _ = model(s1["input_ids"], s1["attention_mask"], p["input_ids"], p["attention_mask"], s2["input_ids"], s2["attention_mask"])
        pred = torch.argmax(logits, dim=-1).item()
        stance = ["Neutral", "Against", "Favor"][pred]
        print(f"[INFO] Prediction: {stance}")
        return stance

# ======================== PLOTTING ======================== #


# Example: taking last epoch F1 score for each target
last_epoch_f1 = {target: scores[-1] for target, scores in f1_scores_by_target.items()}

plt.figure(figsize=(10, 6))
plt.bar(last_epoch_f1.keys(), last_epoch_f1.values(), color='skyblue')
plt.title("F1 Score per Target (Last Epoch)")
plt.xlabel("Target")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()





from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ======================== TESTING & CONFUSION MATRIX ======================== #
print("[INFO] Evaluating on test data...")

# Load test data
print("[INFO] Loading test dataset...")
test_dataset = StanceDataset("C:\\Users\\HP\\Desktop\\ezsd\\final_test_dataset.csv", tokenizer)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
print(f"[INFO] Loaded {len(test_dataset)} test samples.")

model.eval()
all_preds, all_labels = []

# Define label maps
id2label = {0: "NEUTRAL", 1: "AGAINST", 2: "FAVOR"}
label2id = {v: k for k, v in id2label.items()}
label_order = ["NEUTRAL", "AGAINST", "FAVOR"]

print("[INFO] Beginning inference...")
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        print(f"[INFO] Processing batch {i+1}/{len(test_loader)}...")
        s1_input_ids, s1_attention_mask, s2_input_ids, s2_attention_mask, labels, p_input_ids, p_attention_mask, _ = batch

        # Move to device
        s1_input_ids, s1_attention_mask = s1_input_ids.to(device), s1_attention_mask.to(device)
        s2_input_ids, s2_attention_mask = s2_input_ids.to(device), s2_attention_mask.to(device)
        p_input_ids, p_attention_mask = p_input_ids.to(device), p_attention_mask.to(device)
        labels = labels.to(device)

        # Forward pass
        logits, _ = model(s1_input_ids, s1_attention_mask, p_input_ids, p_attention_mask, s2_input_ids, s2_attention_mask)
        preds = torch.argmax(logits, dim=1)

        # Convert IDs to labels
        decoded_preds = [id2label[p.item()] for p in preds]
        decoded_labels = [id2label[l.item()] for l in labels]

        print(f"[DEBUG] Predictions: {decoded_preds}")
        print(f"[DEBUG] Labels:      {decoded_labels}")

        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_labels)

print("[INFO] Inference complete.")
print("[INFO] Computing confusion matrix...")

# Compute confusion matrix with fixed label order
cm = confusion_matrix(all_labels, all_preds, labels=label_order)
print(f"[INFO] Confusion matrix computed:\n{cm}")

# Display the confusion matrix with correct label alignment
print("[INFO] Plotting confusion matrix...")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_order)
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix on Test Data")
plt.show()
print("[INFO] Confusion matrix displayed.")










