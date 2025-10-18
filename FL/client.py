import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import flwr as fl
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score
from imblearn.over_sampling import SMOTE
from project.models import SepsisNet
import os
from sklearn.impute import SimpleImputer



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Load and preprocess dataset
# --------------------------
df = pd.read_csv("Dataset.csv")
df.dropna(subset=["SepsisLabel"], inplace=True)

X = df.drop(columns=["SepsisLabel", "Patient_ID"], errors='ignore')
y = df["SepsisLabel"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# --------------------------
# Apply SMOTE to balance classes
# --------------------------
# Handle NaN values before SMOTE
imputer = SimpleImputer(strategy="median")  # or "mean"
X = imputer.fit_transform(X)

print("[INFO] Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"Before SMOTE: {dict(pd.Series(y).value_counts().sort_index().to_dict())}")
print(f"After SMOTE: {dict(pd.Series(y_res).value_counts().sort_index().to_dict())}")

# --------------------------
# Train/test split after SMOTE
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

full_train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                   torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.long))

# --------------------------
# Client dataset partitioning
# --------------------------
NUM_CLIENTS = int(input("Enter total number of clients: "))
client_id = int(input("Enter client ID (0,1,2,...): "))

# Compute dataset indices for this client
total_samples = len(full_train_dataset)
samples_per_client = total_samples // NUM_CLIENTS
start_idx = client_id * samples_per_client
end_idx = start_idx + samples_per_client if client_id != NUM_CLIENTS - 1 else total_samples

train_subset = Subset(full_train_dataset, list(range(start_idx, end_idx)))

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --------------------------
# Flower client definition
# --------------------------
class SepsisClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Compute class weights for imbalance (even after SMOTE, small differences may remain)
        labels = [y for _, y in self.train_loader.dataset]
        class_counts = torch.bincount(torch.tensor(labels))
        self.class_weights = (1.0 / class_counts.float()).to(DEVICE)
        print(f"[CLIENT {client_id}] Class weights: {self.class_weights}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(),
                              [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss(weight=self.class_weights)

        for epoch in range(5):  # 3 local epochs per round
            total_loss = 0.0
            for x, y in self.train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss = crit(self.model(x), y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            print(f"[CLIENT {client_id}] Epoch {epoch+1}, Loss: {total_loss/len(self.train_loader):.4f}")

        print(f"[CLIENT {client_id}] Finished local training step")
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        crit = nn.CrossEntropyLoss()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = self.model(x)
                total_loss += crit(out, y).item()
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        accuracy = correct / total
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(f"[CLIENT {client_id}] Eval — Acc: {accuracy:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return total_loss / len(self.test_loader), total, {
            "accuracy": accuracy,
            "recall": recall,
            "f1": f1,
        }

# --------------------------
# Start client
# --------------------------
if __name__ == "__main__":
    model = SepsisNet()
    client = SepsisClient(model, train_loader, test_loader)
    print(f"[CLIENT {client_id}] Connecting to server...")
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
