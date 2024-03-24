'''
    This code is borrowed from https://github.com/marcojira/fld
    Thanks for their great work
'''


import torch
import torchvision.transforms as transforms
from KEN.features.ImageFeatureExtractor import ImageFeatureExtractor
from torchvision.models import inception_v3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InceptionFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None, logger=None):
        self.name = "inception"

        super().__init__(save_path, logger)

        self.features_size = 2048
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (299, 299), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self.model = inception_v3(weights='IMAGENET1K_V1')
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(DEVICE)
        self.model.eval()
        return
    
    def get_feature_batch(self, img_batch: torch.Tensor):
        return self.model(img_batch)
