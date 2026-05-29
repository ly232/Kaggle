import kagglehub

import pandas as pd
import os
import duckdb
import torch

from data_processing import to_tensors
from sklearn.model_selection import train_test_split
from training import HyperparameterTuner, SurvivalModel, Trainer

NUM_EPOCHS = 1000

path = kagglehub.competition_download("titanic")
train_df = pd.read_csv(os.path.join(path, "train.csv"))
test_df = pd.read_csv(os.path.join(path, "test.csv"))


#
# Data explorations.
#

# print(train_df.shape)  # (891, 12)
# print(test_df.shape)  # (418, 11)
print(train_df.head())

survival_counts = duckdb.sql("""
SELECT Survived, COUNT(*) AS count FROM train_df GROUP BY ALL;
""").df()
print(survival_counts)

pclass_counts = duckdb.sql("""
SELECT Pclass, COUNT(*) AS count FROM train_df GROUP BY ALL;
""").df()
print(pclass_counts)

sex_counts = duckdb.sql("""
SELECT Sex, COUNT(*) AS count FROM train_df GROUP BY ALL;
""").df()
print(sex_counts)

age_distribution = duckdb.sql("""
SELECT
    approx_quantile(AGE, 0.10) AS p10_age,
    approx_quantile(AGE, 0.20) AS p20_age,
    approx_quantile(AGE, 0.30) AS p30_age,
    approx_quantile(AGE, 0.40) AS p40_age,
    approx_quantile(AGE, 0.50) AS p50_age,
    approx_quantile(AGE, 0.60) AS p60_age,
    approx_quantile(AGE, 0.70) AS p70_age,
    approx_quantile(AGE, 0.80) AS p80_age,
    approx_quantile(AGE, 0.90) AS p90_age,
    approx_quantile(AGE, 0.99) AS p99_age
FROM train_df;
""").df()
print(age_distribution)

cabin_counts = duckdb.sql("""
SELECT Cabin, COUNT(*) AS count FROM train_df GROUP BY ALL order by count DESC;
""").df()
print(cabin_counts)


X_train, y_train = to_tensors(train_df)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,  # labels are imbalanced.
)

tuner = HyperparameterTuner(X_tr, y_tr, X_val, y_val)
tuner.tune()

model = SurvivalModel(X_train.shape[1], tuner.best_params["hidden_dim"])
trainer = Trainer(model, tuner.best_params["lr"])
accuracy, losses = trainer.train(X_tr, y_tr, X_val, y_val)
print(f"Validation Accuracy: {accuracy:.4f}")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "best_params": tuner.best_params,
        "accuracy": accuracy,
        "losses": losses,
    },
    "best_model.pt",
)

# plot the training loss curve.
import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("training_loss_curve.png")
plt.close()


# Now run inference against test set.
print("sample test df:")
print(test_df.head())
X_test, _ = to_tensors(test_df)
with torch.no_grad():
    print(X_train.shape)
    print(X_test.shape)
    test_outputs = model(X_test)
    test_predictions = (test_outputs >= 0.5).float()
    # Save the predictions to a CSV file for submission.
    submission_df = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": test_predictions.squeeze().int().tolist(),
        }
    )
    submission_df.to_csv("submission.csv", index=False)
