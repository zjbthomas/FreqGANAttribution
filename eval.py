import os
import cv2
import numpy as np
import sys
import argparse
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, f1_score
from tqdm import tqdm
from matplotlib import pyplot as plt
import csv
import random
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F

from torchvision.utils import save_image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

from models.A import *
from utils.pilresize import PILResize
from utils.FCRDCT import *
from utils.tsne import *
from utils.rearrange import *

def read_paths(iut_paths_file, undersampling, subset):
    distribution = dict()
    n_min = None

    if (iut_paths_file.endswith('.csv')): # for Attribution88 CSV file
        # csv -> dataframe -> lists
        df = pd.read_csv(iut_paths_file)
        iut_paths = df['path'].tolist()
        labels = np.array(df['label'].tolist())

        assert len(iut_paths) == len(labels)

        # get parent directory (add it to iut_path later)
        prefix = Path(iut_paths_file).parent.absolute()

        for iut_path, label in zip(iut_paths, labels):
            if (label not in distribution):
                distribution[label] = [os.path.join(prefix, iut_path)]
            else:
                distribution[label].append(os.path.join(prefix, iut_path))

    else:
        with open(iut_paths_file, 'r') as f:
            lines = f.readlines()
            for l in lines:
                parts = l.rstrip().split('\t')
                iut_path = parts[0]
                label = int(parts[1])

                if (subset and subset not in parts[0]):
                    continue
                
                # add to distribution
                if (label not in distribution):
                    distribution[label] = [iut_path]
                else:
                    distribution[label].append(iut_path)

    for label in distribution:
        if (n_min is None or len(distribution[label]) < n_min):
            n_min = len(distribution[label])

    # undersampling
    iut_paths_labels = []

    for label in distribution:
        ll = distribution[label]

        if (undersampling == 'all'):
            for i in ll:
                iut_paths_labels.append((i, label))
        elif (undersampling == 'min'):
            picked = random.sample(ll, n_min)
            
            for p in picked:
                iut_paths_labels.append((p, label))
        else:
            print('Unsupported undersampling method {}!'.format(undersampling))
            sys.exit()

    return iut_paths_labels

def save_cm(y_true, y_pred, save_path):
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)

def get_label_strs(n_classes, iut_paths_file, test_set = True):
    label_strs = [''] * n_classes
    if (not iut_paths_file.endswith('.csv')):
        #label_strs = ['real', 'biggan', 'crn', 'cyclegan', 'deepfake', 'gaugan', 'imle', 'progan', 'san', 'seeingdark', 'stargan', 'stylegan', 'stylegan2', 'whichfaceisreal']
        label_strs = ['real', 'ProGAN', 'MMDGAN', 'SNGAN', 'InfoMax-GAN']
    else:
        # csv -> dataframe -> lists
        df = pd.read_csv(iut_paths_file)
        iut_paths = df['path'].tolist()
        labels = np.array(df['label'].tolist())

        assert len(iut_paths) == len(labels)

        for iut_path, label in zip(iut_paths, labels):
            label_strs[label] = iut_path.split('/')[1] if test_set else iut_path.split('/')[0]

    for s in label_strs:
        if (s == ''):
            print('Unrecognized label strings!')
            sys.exit()

    return label_strs

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument("--iut_paths_file", type=str, default="/dataset/iut_files.txt", help="path to the file with paths for image under test") # each line of this file should contain "/path/to/image.ext i", i is an integer represents classes
    parser.add_argument("--image_size", type=int, default=1024, help="size of images")
    parser.add_argument("--crop_size", type=int, default=256, help="size of cropped images")

    parser.add_argument("--subset", type=str, help="evaluation on certain subset")
    parser.add_argument("--undersampling", type=str, default='min', choices=['all', 'min'])

    parser.add_argument("--n_classes", type=int, default=14, help="number of classes")

    parser.add_argument('--out_dir', type=str, default='out')
    
    parser.add_argument('--load_path', type=str, help='path to the pretrained model', default="checkpoints/model.pth")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if (args.model == 'levefreq'):
    #     args.log_dct = 'yes'

    model = Attributor(args.crop_size, args.n_classes).cuda()

    if args.load_path != None and os.path.exists(args.load_path):
        print('Load pretrained model: {}'.format(args.load_path))
        model.load_state_dict(torch.load(args.load_path, map_location=device))
    else:
        print("%s not exist" % args.load_path)
        sys.exit()

    # no training
    model.eval()

    # read paths for data
    if not os.path.exists(args.iut_paths_file):
        print("%s not exists, quit" % args.iut_paths_file)
        sys.exit()

    if (args.subset):
        print("Evaluation on subset {}".format(args.subset))

    iut_paths_labels = read_paths(args.iut_paths_file, args.undersampling, args.subset)

    print("Eval set size is {}!".format(len(iut_paths_labels)))

    # get labels
    label_strs = get_label_strs(args.n_classes, args.iut_paths_file)

    # create/reset output folder
    print("Predicted maps will be saved in :%s" % args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    if (args.subset is None):
        os.makedirs(os.path.join(args.out_dir, 'images'), exist_ok=True)

        for ix in range(args.n_classes):
            os.makedirs(os.path.join(args.out_dir, 'images', label_strs[ix]), exist_ok=True)

    # save paths
    if (args.undersampling == 'min'):
        save_path = os.path.join(args.out_dir, 'paths_file_eval.txt')
        with open(save_path, 'w') as f:
            for (iut_path, label) in iut_paths_labels:
                f.write(iut_path + '\t' + str(label) + '\n')

        print('Eval paths file saved to %s' % (save_path))

    # csv
    if (args.subset is None):
        f_csv = open(os.path.join(args.out_dir, 'pred.csv'), 'w', newline='')
        writer = csv.writer(f_csv)

        header = ['Image', 'Pred', 'True', 'Correct']
        writer.writerow(header)

    # transforms
    transform_dct = A.Compose([
        A.Normalize(mean=0.0, std=1.0), 
        ToTensorV2(),
        Rearrange(args.crop_size),
        DCT(p = 1.0, log=True, factor=1),
        Flatten(),
    ])

    # init t-SNE
    tsne = TSNE(label_strs)
    
    ## prediction
    y_pred = []
    y_true = []

    det_pred = []
    det_true = []

    ws = []

    for ix, (iut_path, lab) in enumerate(tqdm(iut_paths_labels, mininterval = 60)):
        try:
            img = cv2.cvtColor(cv2.imread(iut_path), cv2.COLOR_BGR2RGB)
        except:
            print('Failed to load image {}'.format(iut_path))
            continue
        if (img is None):
            print('Failed to load image {}'.format(iut_path))
            continue

        # DCT
        dct_tensor = transform_dct(image = img)['image'].to(device)

        C, H, W = dct_tensor.shape
        reshaped = torch.reshape(dct_tensor, (C // 3, 3, H, W))

        # prediction
        with torch.no_grad():

            out_att_sum = None

            for p in reshaped:
                out_att = model(p.unsqueeze(0))

                if (out_att_sum is None):
                    out_att_sum = out_att
                else:
                    out_att_sum += out_att

            out_att_avg = out_att_sum / (C // 3)

        y = torch.argmax(F.log_softmax(out_att_avg, dim = 1), dim = 1)
        
        y_pred.append(y.item())
        y_true.append(lab)

        det_pred.append(0 if y.item() == 0 else 1)
        det_true.append(0 if lab == 0 else 1)
        
        # for t-SNE
        #with torch.no_grad():
        #    w = model.forward_features(dct_tensor.unsqueeze(0))
        #ws.append(np.array(w.squeeze(0).detach().cpu()))

        # write to csv
        if (args.subset is None):
            row = [iut_path, y.item(), lab, y.item() == lab]
            writer.writerow(row)

    ## accuracy
    print("acc%s: %.4f" % ((' (' + args.subset + ')' if args.subset else ''), accuracy_score(y_true, y_pred)))

    ## confusion matrix
    #save_path = os.path.join(args.out_dir, 'cm' + ('_' + args.subset if args.subset else '') + '.png')
    #save_cm(y_true, y_pred, save_path)

    ## accuracy (det)
    print("det acc%s: %.4f" % ((' (' + args.subset + ')' if args.subset else ''), accuracy_score(det_true, det_pred)))

    print("det f1 acc%s: %.4f" % ((' (' + args.subset + ')' if args.subset else ''), f1_score(det_true, det_pred)))

    ## confusion matrix (det)
    #save_path = os.path.join(args.out_dir, 'cm' + ('_' + args.subset if args.subset else '') + '_det.png')
    #save_cm(det_true, det_pred, save_path)
    
    ## t-SNE
    #save_path = os.path.join(args.out_dir, 'tsne' + ('_' + args.subset if args.subset else '') + '.png')
    #plt = tsne.gen_tsne_plt(ws, y_true, args.n_classes)
    #plt.savefig(save_path, dpi=300)

    if (args.subset is None): f_csv.close()