# from dags.train import train
from dags.preprocessing.Dataset import Dataset
# from dags.preprocessing.SlidingWIndowCV import SlidingWindowSplitCV
# import pandas as pd

# train_data = Dataset('/kaggle/input/competitive-data-science-predict-future-sales', train=True)
train_data = Dataset('test_data', train=True)
# test_data = Dataset('test_data', train=False)
# cv = SlidingWindowSplitCV(3, 2)

# model = train(train_data, cv)
# model.save_model('model.cbm')

