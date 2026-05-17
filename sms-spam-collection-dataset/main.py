import kagglehub
from kagglehub import KaggleDatasetAdapter

import kagglehub
import util

# Load the latest version
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    handle="uciml/sms-spam-collection-dataset",
    path="spam.csv",
    pandas_kwargs={"encoding": "latin1"},
    # Provide any additional arguments like
    # sql_query or pandas_kwargs. See the
    # documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)
df = util.clean_sms_spam_collection_dataset(df)

print("== First 5 records ==\n", df.head())
