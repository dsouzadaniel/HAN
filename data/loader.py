# External Libraries
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class YelpLoader(Dataset):
    def __init__(self, path_to_csv):
        self.data_tuples = []
        df = pd.read_csv(path_to_csv)
        df['text'] = df['text'].apply(str.lower)
        df['stars'] = df['stars'].apply(lambda x: x - 1)
        self.data_tuples = [(r['text'], r['stars']) for _, r in df.iterrows()]
        self.ctgry = sorted(df['stars'].unique())

    def __getitem__(self, index):
        return self.data_tuples[index]

    def __len__(self):
        return len(self.data_tuples)
