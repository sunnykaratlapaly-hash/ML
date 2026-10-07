
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "esports_match_outcome_prediction_dataset.csv"
MODEL_DIR = BASE / "models"

app = Flask(__name__)

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

def load_artifacts():
    with open(MODEL_DIR / "preprocessor.pkl", "rb") as f:
        prep = pickle.load(f)
    state = torch.load(MODEL_DIR / "model.pt", map_location="cpu")
    model = OutcomeNet(state["input_dim"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    return prep, model

PREP, MODEL = load_artifacts()
df = pd.read_csv(DATA)

NUMERIC = [
    "kills", "deaths", "assists", "gold_difference",
    "objective_score", "team_rating", "opponent_rating",
    "recent_wins", "recent_losses", "map_advantage",
    "match_duration_minutes"
]
CATEGORICAL = ["team", "opponent", "map", "match_mode"]

def make_features(form):
    row = pd.DataFrame([form])
    for c in NUMERIC:
        row[c] = pd.to_numeric(row[c])
    x_num = row[NUMERIC].astype(float).values
    x_num = PREP["scaler"].transform(x_num)
    x_cat = pd.get_dummies(row[CATEGORICAL], dtype=float)
    x_cat = x_cat.reindex(columns=PREP["dummy_columns"], fill_value=0)
    x = np.hstack([x_num, x_cat.values]).astype(np.float32)
    return x

@app.route("/")
def index():
    return render_template(
        "index.html",
        teams=sorted(set(df["team"]) | set(df["opponent"])),
        maps=sorted(df["map"].unique()),
        modes=sorted(df["match_mode"].unique()),
        total=len(df),
        wins=int((df["outcome"] == "Win").sum()),
        losses=int((df["outcome"] == "Loss").sum())
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form.to_dict()
        x = make_features(form)
        with torch.no_grad():
            logit = MODEL(torch.tensor(x)).item()
            win_prob = float(torch.sigmoid(torch.tensor(logit)).item())
        outcome = "Win" if win_prob >= 0.5 else "Loss"
        confidence = win_prob if outcome == "Win" else 1 - win_prob
        return render_template("result.html", outcome=outcome,
                               win_prob=round(win_prob * 100, 2),
                               loss_prob=round((1-win_prob) * 100, 2),
                               confidence=round(confidence * 100, 2),
                               form=form)
    except Exception as e:
        return render_template("error.html", error=str(e)), 400

@app.route("/dataset")
def dataset():
    cols = df.columns.tolist()
    records = df.head(100).to_dict(orient="records")
    return render_template("dataset.html", columns=cols, records=records, total=len(df))

@app.route("/api/stats")
def stats():
    return jsonify({
        "total_matches": int(len(df)),
        "wins": int((df.outcome == "Win").sum()),
        "losses": int((df.outcome == "Loss").sum()),
        "win_rate": round((df.outcome == "Win").mean() * 100, 2),
        "avg_kills": round(df.kills.mean(), 2),
        "avg_objective_score": round(df.objective_score.mean(), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
