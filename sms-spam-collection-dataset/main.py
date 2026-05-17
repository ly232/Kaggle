import kagglehub
from kagglehub import KaggleDatasetAdapter

import kagglehub
import naive_bayes

# Load the latest version
raw_df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    handle="uciml/sms-spam-collection-dataset",
    path="spam.csv",
    pandas_kwargs={"encoding": "latin1"},
    # Provide any additional arguments like
    # sql_query or pandas_kwargs. See the
    # documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("== First 5 records ==\n", raw_df.head())

# Naive Bayes Classifier
naive_bayes_classifier = naive_bayes.NaiveBayesClassifier(raw_df)

print("== First 5 records from NB ==\n", naive_bayes_classifier.df.head())

sample_token = naive_bayes_classifier.df["tokens"].iloc[0]

print(
    'value counts for "label" column:\n',
    naive_bayes_classifier.df["label"].value_counts(),
)

print("== STARTING EVALUATION for Naive Bayes Classifier ==")
naive_bayes_classifier.eval()
