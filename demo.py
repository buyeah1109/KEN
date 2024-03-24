from KEN.metric.KEN import KEN_Evaluator
from KEN.datasets.ImageFilesDataset import ImageFilesDataset
import torch

if __name__ == '__main__':
    torch.hub.set_dir('[path]/torch_hub')

    KEN = KEN_Evaluator(logger_path='./logs', batchsize=128, sigma=15, eta=1, num_samples=5000, result_name='afhq_imgnetdog')

    novel_path = '[path]/datasets/afhq512/data/train'
    ref_path = '[path]/datasets/dogs/Images'

    novel_dataset = ImageFilesDataset(novel_path)
    ref_dataset = ImageFilesDataset(ref_path, extension="jpg")

    KEN.set_feature_extractor('dinov2', save_path='./save')
    KEN.compute_KEN_with_datasets(novel_dataset, ref_dataset, retrieve_mode=True)
