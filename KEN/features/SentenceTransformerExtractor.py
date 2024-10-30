from typing import Tuple
from sentence_transformers import SentenceTransformer

from KEN.features.TextFeatureExtractor import TextFeatureExtractor


class SentenceTransformerExtractor(TextFeatureExtractor):
    def __init__(self, model, save_path=None, logger=None):
        self.name = model.split("/")[-1]
        super().__init__(save_path, logger)

        self.model = SentenceTransformer(model)
        self.features_size = self.model.get_sentence_embedding_dimension()

    def get_feature_batch(self, text_batch: Tuple[str, ...]):
        return self.model.encode(text_batch, convert_to_tensor=True)
