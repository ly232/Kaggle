# Spam Detection with Naive Bayes Classifier

Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Overview

This project implements a simple spam detector using a Naive Bayes classifier.
The input is an SMS message, and the output is one of two labels: `spam` or `ham`.
The model estimates conditional probabilities over discrete token features.

## Theory

Let `D = {(x[i], y[i])}` be the collected training data. This defines an estimated
joint distribution `p_est(x, y)`. If the samples are independently collected, and
the dataset is large enough, `p_est(x, y)` can approximate the true distribution
`p(x, y)`.

The goal of a classifier is to estimate `p(y | x)`. A naive direct approach is:

```python
def eval(x: Input, ds: DataSet) -> Class:
    probabilities_by_class = {}
    for c in Class:
        probabilities_by_class[c] = (
            ds.count(input==x and output==c) / ds.count(input==x)
        )
    return argmax(probabilities_by_class)
```

This works only when every `x` appears in the training data. Otherwise the
calculation can become `0 / 0`. For high-dimensional inputs, exact matches are
very unlikely.

Instead, we use Bayes' rule:

```text
p(y | x) = p(x | y) * p(y) / p(x)
```

Key points:

* `p(x | y)` can be decomposed into `product(p(x_i | y))`. This assumes the
  input features `x_i` are independent given the class. This assumption is not
  always true, but it works well in practice for tasks such as spam detection.
* `p(y)` is the class prior: `ds.count(output == y) / ds.count(*)`.
* `p(x)` is the normalizing constant `sum_y p(x | y) * p(y)`. For inference,
  it does not affect the class argmax, so we can ignore it.

To compute `p(x_i | y)`, transform the training data from `D = {(x, y)}` to
pairs `(x_i, y)`, then estimate:

```text
p(x_i | y) = ds.count(input == x_i and output == y) / ds.count(output == y)
```

At inference time, choose the label `y` that maximizes `p(x | y) * p(y)`. For
numerical stability, use logarithms:

```text
log p(x | y) = sum_i log p(x_i | y)
```

## Connecting to Spam Detection

For SMS classification, each message is a sentence string and the label is either
`spam` or `ham`.

One way to convert text into features is to tokenize each message and treat each
token as an input feature `x_i`. In this implementation, a GPT-2 tokenizer is
used to transform each SMS into a sequence of token IDs. Each `(token, label)`
pair becomes a training example for the Naive Bayes estimator.

## Laplace Smoothing

The basic frequency estimate is:

```text
ds.count(input == x_i and output == y) / ds.count(output == y)
```

To avoid zero probabilities for rare tokens, we use Laplace smoothing:

```text
(ds.count(input == x_i and output == y) + 1) /
(ds.count(output == y) + num_vocabs)
```

Without smoothing, a token that never appears in a class yields probability 0,
which leads to `log(0) = -inf`. In a long message, a single unseen token can
cause the entire score for that class to become `-inf`.

For this dataset, adding Laplace smoothing improved the F1 score from [0.57](https://github.com/ly232/Kaggle/commit/651cb8abe07a5c7d025a4a997eb4a11b2326544d) to
[0.94](https://github.com/ly232/Kaggle/commit/0de7261f8c86e2c997377bac540eb772b7ceaccc).
