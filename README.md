# Privacy-Preserving Federated Learning for Sepsis Prediction

Team no: 24

Collorators: Disha R, Chinmayi Shravani Yellanki

This repository demonstrates **Federated Learning (FL)** for **Sepsis Prediction** using **Flower (FLwr)** and **PyTorch**, including experiments on **privacy-preserving techniques** such as **Differential Privacy (DP)** and **Homomorphic Encryption (HE)**.  

It contains both a **production-ready FL project** and **simulation notebooks** to explore different setups and threats.

---
## Dataset

The dataset used in this project is sourced from Kaggle:

[Prediction of Sepsis Dataset – Kaggle](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis)

To use it:
1. Download the dataset manually from Kaggle.
2. Place the file in the `FL/` folder.

Example:


## Repository Structure

```text
FL/
│
├── server.py             # FL server for orchestrating rounds
├── client.py             # FL client for local training
├── models.py             # SepsisNet model definition
├── evaluate.py           # Evaluate global model
├── requirements.txt      # Python dependencies
├── Dataset.csv           # Optional dataset for testing
│
Notebooks/
├── Central_learning.ipynb # Centralised learning
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

- `Central_learning.ipynb` - simulates centralised learning to compared with federated learning
- `FL_non_iid.ipynb` – simulates FL with **non-IID client data**.  
- `FL_iid.ipynb` – simulates FL with **IID client data**.  
- `gradient_leakage.ipynb` – demonstrates **gradient leakage attacks** and the effect of:
  - **Differential Privacy (DP)**
  - **Homomorphic Encryption (HE)**  

These notebooks are **educational simulations** and allow experimenting with **different privacy setups**.

---

## Requirements

Make sure **Python 3.9+** is installed.  

Install dependencies:

```bash
pip install -r requirements.txt
```

## To Run FL Folder

This project implements **Federated Learning (FL)** using **Flower (FLwr)** locally  

### Steps:

1. **Open `n + 1` terminals**:
   - 1 terminal for the **server**  
   - `n` terminals for **clients** (number of clients you want to run)

2. **Start the Server**:
   - In the first terminal, navigate to the project folder:
     ```bash
     cd path/to/project
     ```
   - Run the server:
     ```bash
     python server.py
     ```
   - **Optional customization in `server.py`**:
     ```python
     num_clients = 3   # Total number of clients
     num_rounds = 5    # Total FL rounds
     ```

3. **Start the Clients**:
   - In each of the `n` terminals, navigate to the project folder:
     ```bash
     cd path/to/project
     ```
   - Run the client:
     ```bash
     python client.py
     ```
   - Enter the prompts:
     ```
     Enter total number of clients: 3
     Enter Client ID: 1         # Unique ID for each client
     
     ```
   - Each client:
     - Loads its dataset portion  
     - Trains the `SepsisNet` model locally  
     - Sends updates to the server  

4. **Evaluate the Final Model**:
   - After training is complete, run:
     ```bash
     python evaluate.py
     ```
   - Enter the global model `.pth` file number to evaluate:
     ```
     Enter global model round to evaluate: 5
     ```
   - Output includes:
     - Accuracy  
     - F1-score  
     - Recall  
     - Confusion Matrix  
     - ROC-AUC (if implemented)
    
## To Run the Notebooks

This repository also contains **Jupyter notebooks** for simulating Federated Learning experiments and privacy scenarios:

- `FL_non_iid.ipynb` – FL with Non-IID client data which uses FedProx algorithm
- `FL_iid.ipynb` – FL with IID client data 
- `gradient_leakage.ipynb` – Experiments with/without Differential Privacy (DP) and with Homomorphic Encryption (HE) to evaluate privacy

### Steps:

1. **Open Jupyter Lab/Notebook**:

- The notebooks are self-contained simulations; no need to run server.py or client.py.
- The datasets you upload are used locally within the notebook.
- All visualizations (metrics, loss curves, ROC-AUC) will be displayed inline in the notebook.
  




