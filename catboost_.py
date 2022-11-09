# %% [markdown]
# from google.colab import drive
# drive.mount('/content/drive/')
#

# %% [markdown]
# %cd drive/MyDrive/FutureSalesPredict
# %ls

# %% [markdown]
# !sudo apt-get install python3.10
# !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
# !sudo update-alternatives --set python3 /usr/bin/python3.10
# !python --version

# %% [markdown]
# !git pull

# %%
from dags.train import train
from dags.preprocessing.Dataset import Dataset
from dags.preprocessing.SlidingWIndowCV import SlidingWindowSplitCV

train_data = Dataset('test_data', train=True, add_item_cartesian_product=True)
cv = SlidingWindowSplitCV(12,2)

_ = train(train_data, cv)
# print(res['models'])

# %%
