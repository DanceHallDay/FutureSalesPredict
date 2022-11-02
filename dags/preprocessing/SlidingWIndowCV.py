import numpy as np
import pandas as pd


class SlidingWindowSplitCV:
    #TODO add docstring
    """Parameters
    -
     - train_size - number of months

     - test_size - number of months
    """
    def __init__(self, train_size: int, test_size: int):

        self.train_size = train_size
        self.test_size = test_size

    def split(self, df: pd.DataFrame, split_column: str):

        df = df.sort_values(by=split_column).reset_index()
        border_indexes = df.loc[df[split_column] !=
                                df[split_column].shift()].index.tolist()

        for i in range(34 - (self.train_size + self.test_size)):
            train_indexes = np.array(
                [
                    *range(
                        border_indexes[i],
                        border_indexes[i + self.train_size]
                    )
                ]
            )
            test_indexes = np.array(
                [
                    *range(
                        border_indexes[i + self.train_size],
                        border_indexes[i + self.train_size + self.test_size]
                    )
                ]
            )
            yield train_indexes, test_indexes
