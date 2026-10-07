
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "esports_match_outcome_prediction_dataset.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

NUMERIC = [
    "kills", "deaths", "assists", "gold_difference",
    "objective_score", "team_rating", "opponent_rating",
    "recent_wins", "recent_losses", "map_advantage",
    "match_duration_minutes"
]
CATEGORICAL = ["team", "opponent", "map", "match_mode"]

class OutcomeNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

df = pd.read_csv(DATA)
y = (df["outcome"] == "Win").astype(np.float32).values

scaler = StandardScaler()
x_num = scaler.fit_transform(df[NUMERIC].astype(float))

x_cat = pd.get_dummies(df[CATEGORICAL], dtype=float)
dummy_columns = x_cat.columns.tolist()
x = np.hstack([x_num, x_cat.values]).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42, stratify=y
)

torch.manual_seed(42)
model = OutcomeNet(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

Xt = torch.tensor(X_train)
yt = torch.tensor(y_train).reshape(-1, 1)

for epoch in range(250):
    model.train()
    optimizer.zero_grad()
    logits = model(Xt)
    loss = criterion(logits, yt)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    pred = (torch.sigmoid(model(torch.tensor(X_test))) >= 0.5).numpy().ravel()
accuracy = float((pred == y_test).mean())
print(f"Test accuracy: {accuracy*100:.2f}%")

with open(MODEL_DIR / "preprocessor.pkl", "wb") as f:
    pickle.dump({"scaler": scaler, "dummy_columns": dummy_columns}, f)

torch.save({
    "input_dim": X_train.shape[1],
    "state_dict": model.state_dict(),
    "test_accuracy": accuracy
}, MODEL_DIR / "model.pt")

print("Model saved.")
