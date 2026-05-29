"""Hyperparameter tuning with Optuna."""

from sklearn.metrics import f1_score

import optuna
import torch
import torch.nn as nn
import tqdm

NUM_EPOCHS = 1000


class SurvivalModel(nn.Module):
    """A simple feedforward neural network for binary classification."""

    def __init__(self, input_dim, hidden_dim):
        super(SurvivalModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class Trainer:
    """Trainer class to handle the training loop."""

    def __init__(self, model, lr):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_tr, y_tr, X_val, y_val):
        loss_fn = nn.BCELoss()

        losses: list[float] = []  # for plotting.
        for _ in tqdm.tqdm(range(NUM_EPOCHS), desc="Training"):
            self.optimizer.zero_grad()
            outputs = self.model(X_tr)
            loss = loss_fn(outputs, y_tr)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        with torch.no_grad():
            preds = (self.model(X_val) >= 0.5).float()
            accuracy = (preds == y_val).float().mean().item()
            f1 = f1_score(y_val.squeeze().numpy(), preds.squeeze().numpy())

        return f1, accuracy, losses


class HyperparameterTuner:
    """Hyperparameter tuning with Optuna."""

    def __init__(self, X_tr, y_tr, X_val, y_val):
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_val = X_val
        self.y_val = y_val
        self.best_params = None

    def objective(self, trial: optuna.Trial) -> float:
        hidden_dim = trial.suggest_int("hidden_dim", 8, 128, log=True)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        model = SurvivalModel(self.X_tr.shape[1], hidden_dim)
        trainer = Trainer(model, lr)
        f1, accuracy, _ = trainer.train(self.X_tr, self.y_tr, self.X_val, self.y_val)
        return accuracy

    def tune(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100)
        print("Best value: ", study.best_value)
        print("Best hyperparameters: ", study.best_params)
        self.best_params = study.best_params
