import os
import torch
from openai import AzureOpenAI
from typing import Tuple
from dotenv import load_dotenv

from KEN.features.TextFeatureExtractor import TextFeatureExtractor


class OpenAIExtractor(TextFeatureExtractor):
    def __init__(self, model, save_path=None, logger=None):
        self.name = model
        super().__init__(save_path, logger)

        load_dotenv()

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.features_size = 768

    def get_feature_batch(self, text_batch: Tuple[str, ...]):
        return torch.tensor(
            [
                x.embedding
                for x in self.client.embeddings.create(
                    input=list(text_batch),
                    model=self.name,
                    dimensions=self.features_size,
                ).data
            ]
        )
