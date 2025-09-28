import os
import random
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
from utils.cnorm import *
from utils.pilresize import *
from utils.FCRDCT import *
from utils.rearrange import *

class AttributorDataset(Dataset):

    def sampling(self, distribution, n_max):
        if self.n_c_samples is None:
            self.n_c_samples = n_max

        for label in distribution:
            ll = distribution[label]
            n_list = len(ll)

            if (n_list >= self.n_c_samples):
                # undersampling
                new_list = random.sample(ll, self.n_c_samples)

            else:
                # oversampling
                new_list = []

                for _ in range(self.n_c_samples // n_list):
                    for i in ll:
                        new_list.append(i)

                new_list.extend(random.sample(ll, self.n_c_samples % n_list))

            random.shuffle(new_list)
            distribution[label] = new_list
        
        return distribution

    def __init__(self, global_rank, iut_paths_file, id, image_size, crop_size, quality, n_c_samples = None, val = False, test = False):

        if (val):
            set_name = 'val'
        elif (test):
            set_name = 'test'
        else:
            set_name = 'train'

        self.n_c_samples = n_c_samples
        
        self.train = not val and not test
        self.test = not val and test

        self.save_path = 'cond_paths_file_' + str(id) + '_' + set_name + '.txt'

        self.distribution = dict()

        if (iut_paths_file.endswith('.csv')): # for Attribution88 CSV file
            # csv -> dataframe -> lists
            df = pd.read_csv(iut_paths_file)
            iut_paths = df['path'].tolist()
            m_ls = np.array(df['label'].tolist())
            s_ls = np.array(df['dlabel'].tolist())

            assert len(iut_paths) == len(m_ls) == len(s_ls)

            # get parent directory (add it to iut_path later)
            prefix = Path(iut_paths_file).parent.absolute()

            # build self.distribution
            n_max = 0

            for iut_path, ml, sl in zip(iut_paths, m_ls, s_ls):
                if (ml not in self.distribution):
                    self.distribution[ml] = [(os.path.join(prefix, iut_path), sl)]
                else:
                    self.distribution[ml].append((os.path.join(prefix, iut_path), sl))

                if (len(self.distribution[ml]) > n_max):
                    n_max = len(self.distribution[ml])
            
            self.distribution = self.sampling(self.distribution, n_max)

            # TODO: save final 

        else: # for self-constructed TXT file
            if ('cond' not in iut_paths_file):
                n_max = 0

                with open(iut_paths_file, 'r') as f:
                    lines = f.readlines()
                    for l in lines:
                        parts = l.rstrip().split('\t')
                        iut_path = parts[0]
                        ml = int(parts[1])
                        sl = int(parts[2])

                        # add to distribution
                        if (ml not in self.distribution):
                            self.distribution[ml] = [(iut_path, sl)]
                        else:
                            self.distribution[ml].append((iut_path, sl))

                        if (len(self.distribution[ml]) > n_max):
                            n_max = len(self.distribution[ml])

                self.distribution = self.sampling(self.distribution, n_max)

                # save final 
                if (global_rank == 0):
                    with open(self.save_path, 'w') as f:
                        for ml in self.distribution:
                            ll = self.distribution[ml]

                            for (p, sl) in ll:
                                f.write(p + '\t' + str(ml) + '\t' + str(sl) + '\n')

                    print('Final paths file (%s) for %s saved to %s' % (set_name, str(id), self.save_path))

            else:
                print('Read from previous saved paths file %s' % (iut_paths_file))

                with open(iut_paths_file, 'r') as f:
                    lines = f.readlines()
                    for l in lines:
                        parts = l.rstrip().split('\t')
                        iut_path = parts[0]
                        ml = int(parts[1])
                        sl = int(parts[2])

                        if (ml not in self.distribution):
                            self.distribution[ml] = [(iut_path, sl)]
                        else:
                            self.distribution[ml].append((iut_path, sl))

        # ----------
        #  TODO: Transforms for data augmentation (more augmentations should be added)
        # ----------  
        assert quality[0] <= quality[1]

        quality_upper = 99 if quality[1] == 100 else quality[1]

        if (quality[1] == 100):
            p = 1.0 / (quality[1] - quality[0] + 1)
        else:
            p = 1.0

        self.transform_train = A.Compose([
            A.ImageCompression(quality_lower=quality[0], quality_upper=quality_upper, p=p),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.GaussNoise(var_limit=(10.0, 50.0)),
            # mandatory - pre
            A.Normalize(mean=0.0, std=1.0), 
            # basic
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # mandatory - post
            ToTensorV2(),
            Rearrange(crop_size),
            DCT(p = 1.0, log=True, factor=1),
            RandomPick(),
        ])

        val_test_quality = (quality[0] + quality[1]) // 2
        
        self.transform_val = A.Compose([
            A.ImageCompression(quality_lower=val_test_quality, quality_upper=val_test_quality, always_apply=True, p=1.0),
            A.GaussianBlur(blur_limit=(5,5), always_apply=True, p=1.0),
            A.GaussNoise(var_limit=(30.0,30.0), always_apply=True, p=1.0),
            # mandatory - pre
            A.Normalize(mean=0.0, std=1.0),
            # mandatory - post
            ToTensorV2(),
            Rearrange(crop_size),
            DCT(p = 1.0, log=True, factor=1),
            Flatten(),
        ])

        self.transform_test = A.Compose([
            # mandatory - pre
            A.Normalize(mean=0.0, std=1.0),
            # mandatory - post
            ToTensorV2(),
            Rearrange(crop_size),
            DCT(p = 1.0, log=True, factor=1),
            Flatten(),
        ])

    def __getitem__(self, item):
        # ----------
        # Read images
        # ----------
        images = []
        m_ls = []
        s_ls = []

        for ml in self.distribution:
            ll = self.distribution[ml]

            (iut_filename, sl) = ll[item]
            try:
                iut = cv2.cvtColor(cv2.imread(iut_filename), cv2.COLOR_BGR2RGB)
            except:
                print('Failed to load image {}'.format(iut_filename))
                return None
            
            if (iut is None):
                print('Failed to load image {}'.format(iut_filename))
                return None

            # ----------
            # Apply transform
            # ----------
            if (self.train):
                iut = self.transform_train(image = iut)['image']
            elif (self.test):
                iut = self.transform_test(image = iut)['image']
            else:
                iut = self.transform_val(image = iut)['image']

            images.append(iut)
            m_ls.append(ml)
            s_ls.append(sl)

        return torch.stack(images), torch.LongTensor(m_ls), torch.LongTensor(s_ls)

    def __len__(self):
        return len(list(self.distribution.values())[0])
