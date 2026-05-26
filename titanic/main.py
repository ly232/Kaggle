import kagglehub

import pandas as pd
import os
import duckdb
import torch
import tqdm

from data_processing import to_tensors
from model import SurvivalClassifier

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

model = SurvivalClassifier(input_dim=X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCELoss()

# training loop
losses: list[float] = []  # for plotting.
for _ in tqdm.tqdm(range(NUM_EPOCHS)):
    # Note this is crucial to avoid optimizer from updating
    # weights with accumulated gradients. We almost always want
    # to do this.
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

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
