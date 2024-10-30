from KEN.metric.KEN import KEN_Evaluator
from KEN.datasets.TextDataset import TextDataset

if __name__ == "__main__":
    KEN = KEN_Evaluator(
        logger_path="./logs",
        batchsize=128,
        sigma=10,
        eta=1,
        num_samples=1000,
        result_name="facts-football_francehistory",
    )

    novel_path = "~/dataset-facts/test.csv"
    ref_path = "~/dataset-facts/reference.csv"

    novel_dataset = TextDataset(novel_path)
    ref_dataset = TextDataset(ref_path)

    KEN.set_feature_extractor(
        "sentence-transformer", "bert-base-nli-mean-tokens", save_path="./save"
    )
    KEN.compute_KEN_with_datasets(
        novel_dataset,
        ref_dataset,
        retrieve_mode=True,
        cholesky_acceleration=True,
    )
