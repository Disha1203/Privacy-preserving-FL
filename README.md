# Privacy-Preserving Federated Learning for Sepsis Prediction

This repository demonstrates **Federated Learning (FL)** for **Sepsis Prediction** using **Flower (FLwr)** and **PyTorch**, including experiments on **privacy-preserving techniques** such as **Differential Privacy (DP)** and **Homomorphic Encryption (HE)**.  

It contains both a **production-ready FL project** and **simulation notebooks** to explore different setups and threats.

---


## 📁 Repository Structure

```text
project/
│
├── server.py             # FL server for orchestrating rounds
├── client.py             # FL client for local training
├── models.py             # SepsisNet model definition
├── evaluate.py           # Evaluate global model
├── requirements.txt      # Python dependencies
├── Dataset.csv           # Optional dataset for testing
│
notebooks/
├── FL_non_iid.ipynb       # Simulate FL with Non-IID data
├── FL_iid.ipynb           # Simulate FL with IID data
├── gradient_leakage.ipynb # Experiments with/without DP and with HE

```
---

## How the FL Project Works

### 1. Federated Learning Pipeline

- **Server** (`server.py`) orchestrates training rounds, aggregates client updates.
- **Clients** (`client.py`) train locally on their private data and send model parameters.
- **Global model** is saved after each round (`logs/global_model_roundX.pth`).
- **Evaluation** (`evaluate.py`) allows testing the global model.

---

### 2. Simulation Notebooks

- `FL_non_iid.ipynb` – simulates FL with **non-IID client data**.  
- `FL_iid.ipynb` – simulates FL with **IID client data**.  
- `gradient_leakage.ipynb` – demonstrates **gradient leakage attacks** and the effect of:
  - **Differential Privacy (DP)**
  - **Homomorphic Encryption (HE)**  

These notebooks are **educational simulations** and allow experimenting with **different privacy setups**.

---

## 📦 Requirements

Make sure **Python 3.9+** is installed.  

Install dependencies:

```bash
pip install -r requirements.txt


