import torch
from argparse import ArgumentParser, Namespace
from .algorithm_utils import *
from os.path import join
from KEN.features.CLIPFeatureExtractor import CLIPFeatureExtractor
from KEN.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from KEN.features.InceptionFeatureExtractor import InceptionFeatureExtractor
import time
import logging
import sys

def get_logger(filepath='./logs/novelty.log'):
    '''
        Information Module:
            Save the program execution information to a log file and output to the terminal at the same time
    '''

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger

class KEN_Evaluator():
    def __init__(self, logger_path: str, sigma : float, eta : float, result_name: str, num_samples: int = 5000, batchsize: int = 128):
        self.logger_path = logger_path
        self.sigma = sigma
        self.eta = eta
        self.num_samples = num_samples
        self.batchsize = batchsize

        self.current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        self.result_name = '{}_num_{}_sigma_{}_eta_{}'.format(result_name, num_samples, sigma, eta)
        self.save_feats_name = '{}_num_{}'.format(result_name, num_samples)


        self.feature_extractor = None
        self.name_feature_extractor = None
        self.running_logger = None

        self.init_running_logger()
        self.running_logger.info("KEN Evaluator Initialized.")
    
    def init_running_logger(self):
        self.running_logger = get_logger(join(self.logger_path, 'run_{}_{}.log'.format(self.result_name, self.current_time)))
    
    def test(self):
        test_args = Namespace(batchsize=32, sigma=1, eta=1, logger=self.running_logger)

        x = torch.randn((32, 2)) * 0.1
        y = torch.randn((32, 2)) * 0.1 + 10
        KEN_by_eigendecomposition(x, y, test_args)
        KEN_by_cholesky_decomposition(x, y, test_args)
    
    def set_feature_extractor(self, name: str, save_path=None):
        if name.lower() == 'inception':
            self.feature_extractor = InceptionFeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'dinov2':
            self.feature_extractor = DINOv2FeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'clip':
            self.feature_extractor = CLIPFeatureExtractor(save_path, logger=self.running_logger)
        else:
            raise NotImplementedError(
                f"Cannot get feature extractor '{name}'. Expected one of ['inception', 'dinov2', 'clip']"
            )
        self.name_feature_extractor = name.lower()
        self.running_logger.info("Initialized feature-extractor network: {}".format(self.name_feature_extractor))
    
    def compute_KEN_with_datasets(self, 
                                  test_dataset: torch.utils.data.Dataset, 
                                  ref_dataset: torch.utils.data.Dataset, 
                                  cholesky_acceleration=False, 
                                  retrieve_mode = False,
                                  retrieve_mode_from_both_sets = False):
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         eta=self.eta, 
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual='./visuals/modes',
                         num_visual_mode=10,
                         num_img_per_mode=50,
                         resize_img_to=224
        )
        
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}, Eta: {}".format(args.num_samples, args.sigma, args.eta))
        self.running_logger.info('test dataset length: {}'.format(len(test_dataset)))
        self.running_logger.info('ref dataset length: {}'.format(len(ref_dataset)))

        if self.feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default Inception-V3.")
            self.set_feature_extractor(name='inception', logger=self.running_logger)

        with torch.no_grad():
            self.running_logger.info("Calculating test feats:")
            test_feats, test_idxs = self.feature_extractor.get_features_and_idxes(test_dataset, 
                                                                    name = 'test_' + self.save_feats_name, 
                                                                    recompute=False, 
                                                                    num_samples=args.num_samples, 
                                                                    batchsize=args.batchsize)
            
            self.running_logger.info("Calculating ref feats:")
            ref_feats, ref_idxs = self.feature_extractor.get_features_and_idxes(ref_dataset, 
                                                                    name = 'ref_' + self.save_feats_name, 
                                                                    recompute=False, 
                                                                    num_samples=args.num_samples, 
                                                                    batchsize=args.batchsize)
        
        self.running_logger.info("number of test feature: {}, reference feature: {}".format(len(test_feats), len(ref_feats)))

        if retrieve_mode:
            self.running_logger.info('Now retrieving modes by top eigenvectors and calculating KEN score.')

            if retrieve_mode_from_both_sets:
                self.running_logger.info('User select to retrieve samples from both test and ref set.')
                visualize_mode_by_eigenvectors_in_both_sets(test_feats, 
                                                            ref_feats, 
                                                            test_dataset, 
                                                            test_idxs, 
                                                            ref_dataset, 
                                                            ref_idxs, 
                                                            args, 
                                                            absolute=True,
                                                            print_KEN=True)
            else:
                self.running_logger.info('User select to retrieve samples from test set only.')
                visualize_mode_by_eigenvectors(test_feats, 
                                                ref_feats, 
                                                test_dataset, 
                                                test_idxs, 
                                                args,
                                                print_KEN=True)
            
            self.running_logger.info('Finished.')
        
        else:
            self.running_logger.info('Now calculating KEN score')
            if cholesky_acceleration:
                KEN_by_cholesky_decomposition(test_feats, ref_feats, args)
            else:
                KEN_by_eigendecomposition(test_feats, ref_feats, args)
    
    def compute_KEN_with_features(self, 
                                  test_feats: torch.Tensor, 
                                  ref_feats: torch.Tensor, 
                                  cholesky_acceleration=False):
        
        args = Namespace(batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         eta=self.eta, 
                         logger=self.running_logger,
        )
        
        self.running_logger.info("Sigma: {}, Eta: {}".format(args.sigma, args.eta))
        self.running_logger.info('number of test feats: {}'.format(test_feats.shape[0]))
        self.running_logger.info('number of ref feats: {}'.format(ref_feats.shape[0]))
        
        self.running_logger.info('Now calculating KEN score')
        if cholesky_acceleration:
            KEN_by_cholesky_decomposition(test_feats, ref_feats, args)
        else:
            KEN_by_eigendecomposition(test_feats, ref_feats, args)

