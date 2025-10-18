import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score
from project.models import SepsisNet
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Load test dataset
# --------------------------
df = pd.read_csv("archive (1)/Dataset.csv")
df.dropna(subset=["SepsisLabel"], inplace=True)

X = df.drop(columns=["SepsisLabel", "Patient_ID"], errors='ignore').values
y = df["SepsisLabel"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# --------------------------
# Test split
# --------------------------
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# Downsample majority class for balanced evaluation
# --------------------------
X0 = X_test[y_test == 0]
y0 = y_test[y_test == 0]
X1 = X_test[y_test == 1]
y1 = y_test[y_test == 1]

X0_down, y0_down = resample(X0, y0, n_samples=len(y1), random_state=42)
X_bal = np.vstack([X0_down, X1])
y_bal = np.hstack([y0_down, y1])

# Convert to tensors
X_bal_tensor = torch.tensor(X_bal, dtype=torch.float32).to(DEVICE)
y_bal_tensor = torch.tensor(y_bal, dtype=torch.long).to(DEVICE)

# --------------------------
# Load global model
# --------------------------
round_no = int(input("Enter global model round to evaluate: " ))
model_path = f"logs/global_model_round{round_no}.pth"

print(f"\n[INFO] Loading model from {model_path}")
state_dict = torch.load(model_path, map_location=DEVICE)
model = SepsisNet()
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# --------------------------
# Evaluate
# --------------------------
all_labels = []
all_preds = []
all_probs = []

batch_size = 32
with torch.no_grad():
    for i in range(0, len(X_bal_tensor), batch_size):
        X_batch = X_bal_tensor[i:i+batch_size]
        y_batch = y_bal_tensor[i:i+batch_size]
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # class 1 prob
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_labels.extend(y_batch.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# Clean NaN/inf probabilities
if np.isnan(all_probs).any() or np.isinf(all_probs).any():
    print("[WARNING] NaN/inf values detected in probabilities. Cleaning...")
    all_probs = np.nan_to_num(all_probs, nan=0.0, posinf=1.0, neginf=0.0)

# --------------------------
# Metrics
# --------------------------
print("\n========== FINAL GLOBAL MODEL EVALUATION ==========")
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, zero_division=0))

# Compute ROC AUC
if len(np.unique(all_preds)) > 1:
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"\nROC AUC: {roc_auc:.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Global Model Round {round_no}')
    plt.legend(loc="lower right")
    plt.show()
else:
    print("\n[WARNING] Model predicted only one class — ROC AUC not computed.")

print("===================================================")
