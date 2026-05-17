"""Implements a Naive Bayes classifier."""

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from transformers import GPT2Tokenizer

import pandas as pd
import numpy as np
import data_processing


class NaiveBayesClassifier:
    def __init__(self, raw_df: pd.DataFrame):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.df = data_processing.clean_sms_spam_collection_dataset(raw_df)
        self.df = data_processing.tokenize_message(self.tokenizer, self.df)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df["tokens"],
            self.df["label"],
            test_size=0.2,
            random_state=42,
            stratify=self.df["label"],  # labels are imbalanced.
        )

        # Compute label probabilities for the Naive Bayes classifier.
        self.p_spam = self.y_train.value_counts()["spam"] / len(self.df)

        # Compute token probabilities conditioned on labels.
        self.num_spams, self.num_hams = (
            self.y_train.value_counts()["spam"],
            self.y_train.value_counts()["ham"],
        )
        self.token_counts_given_spam = self._count_conditional_tokens(
            self.X_train, self.y_train, label_value="spam"
        )
        self.token_counts_given_ham = self._count_conditional_tokens(
            self.X_train, self.y_train, label_value="ham"
        )

    def predict(self, tokens: list[int]) -> str:
        """Predicts the labels for the input data ('spam' or 'ham')."""
        spam_score, ham_score = (
            self._compute_sum_log_probabilities(tokens, label_value="spam"),
            self._compute_sum_log_probabilities(tokens, label_value="ham"),
        )
        return "spam" if spam_score > ham_score else "ham"

    def eval(self) -> None:
        predictions = self.X_test.apply(lambda tokens: self.predict(tokens))
        print("Classification Report:")
        print(classification_report(self.y_test, predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, predictions))
        print("Accuracy:")
        print(accuracy_score(self.y_test, predictions))
        print("Precision:")
        print(precision_score(self.y_test, predictions, pos_label="spam"))
        print("Recall:")
        print(recall_score(self.y_test, predictions, pos_label="spam"))
        print("F1 Score:")
        print(f1_score(self.y_test, predictions, pos_label="spam"))

    def _compute_sum_log_probabilities(
        self, tokens: list[int], label_value: str
    ) -> float:
        assert label_value in ("spam", "ham")
        token_counts = (
            self.token_counts_given_spam
            if label_value == "spam"
            else self.token_counts_given_ham
        )
        label_count = self.num_spams if label_value == "spam" else self.num_hams
        p_label = self.p_spam if label_value == "spam" else 1 - self.p_spam
        token_counts_given_label = token_counts[token_counts.index.isin(tokens)]
        label_probabilities = token_counts_given_label / label_count
        sum_log_prob = label_probabilities.apply(lambda p: np.log(p)).sum() + np.log(
            p_label
        )
        return sum_log_prob

    def _count_conditional_tokens(
        self, tokens: pd.Series, labels: pd.Series, label_value: str = "spam"
    ) -> pd.Series:
        """Counts the frequency of each token conditioned on the given labels.

        Args:
            tokens: A Series where each value is a list of tokens for a message.
            labels: A Series of labels corresponding to each message.
            label_value: The label value to condition on, 'spam' or 'ham'.

        Returns:
            A Series where the index is the token and the value is the frequency of that token, conditioned on label. Example:
            ```
            tokens
            13     3249
            345    1384
            284    1262
            11     1158
            314    1153
            ```
        """
        #  `tokens[labels == "spam"]``:
        #    Series, index is row number, value is *list* of tokens. Note
        #    both X_train and y_train are pd.Series. They share the same
        #    indexes, which is how we can do boolean indexing with
        #    `tokens[labels == "spam"]`.
        #  `tokens[labels == "spam"].explode()``:
        #    Series, index is row number (may have dups), value is unnested
        #    *single* token.
        #  `tokens[labels == 'spam'].explode().value_counts()`:
        #    Series, index is token, value is frequency of token.
        return tokens[labels == label_value].explode().value_counts()
