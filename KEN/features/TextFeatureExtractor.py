import os
import json
import torch
from tqdm import tqdm

NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextFeatureExtractor:
    def __init__(self, save_path: str | None, logger=None):
        if save_path is None:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(curr_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                save_path = config["feature_cache_path"]
                save_path = os.path.join(save_path, self.name)
        else:
            save_path = os.path.join(save_path, self.name)

        self.save_path = save_path
        self.logger = logger
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        # TO BE IMPLEMENTED BY EACH MODULE
        self.features_size = None

    def get_feature_batch(self, text_batch: torch.Tensor):
        """TO BE IMPLEMENTED BY EACH MODULE"""
        pass

    def get_features_and_idxes(
        self,
        texts: torch.utils.data.Dataset,
        name=None,
        recompute=False,
        num_samples=5000,
        batchsize=128,
    ):
        """
        Gets the features from imgs (a Dataset).
        - name: Unique name of set of images for caching purposes
        - recompute: Whether to recompute cached features
        - num_samples: number of samples
        - batchsize: batch size in computing features
        """
        if self.save_path and name:
            file_path = os.path.join(self.save_path, f"{name}.pt")

            if not recompute:
                if os.path.exists(file_path):
                    load_file = torch.load(file_path, weights_only=True)
                    if self.logger is not None:
                        self.logger.info(
                            "Found saved features and idxes: {}".format(file_path)
                        )
                    return load_file["features"], load_file["idxes"]

        if isinstance(texts, torch.utils.data.Dataset):
            features, idxes = self.get_dataset_features_and_idxes(
                texts, num_samples, batchsize
            )
        else:
            raise NotImplementedError(
                f"Cannot get features from '{type(texts)}'. Expected torch.utils.data.Dataset"
            )

        if self.save_path and name:
            if self.logger is not None:
                self.logger.info("Saving features and idxes to {}".format(file_path))
            torch.save({"features": features, "idxes": idxes}, file_path)

        return features, idxes

    def get_dataset_features_and_idxes(
        self, dataset: torch.utils.data.Dataset, num_samples=5000, batchsize=128
    ):
        size = min(num_samples, len(dataset))
        features = torch.zeros(size, self.features_size)
        idxes = torch.zeros(size)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            drop_last=False,
            num_workers=NUM_WORKERS,
            shuffle=True,
        )

        start_idx = 0
        for txt_batch, _, idx in tqdm(
            dataloader, leave=False, total=int(num_samples // batchsize) + 1
        ):
            feature = self.get_feature_batch(txt_batch)

            # If going to overflow, just get required amount and break
            if size and start_idx + feature.shape[0] > size:
                features[start_idx:] = feature[: size - start_idx]
                break

            features[start_idx : start_idx + feature.shape[0]] = feature
            idxes[start_idx : start_idx + feature.shape[0]] = idx

            start_idx = start_idx + feature.shape[0]

        return features, idxes
