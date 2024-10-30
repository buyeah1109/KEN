import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, path):
        pd.set_option("future.no_silent_downcasting", True)

        self.texts = pd.read_csv(path)
        unique_topics = self.texts.topic.unique().tolist()
        self.texts.topic = self.texts.topic.replace(
            unique_topics, np.arange(len(unique_topics))
        )
        self.texts = self.texts.to_numpy()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        label, text = self.texts[idx]
        return text, label, idx

    def get_class(self, idx):
        return self.texts[idx][1]
