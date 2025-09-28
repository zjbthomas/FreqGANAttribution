import argparse
import os
import sys
from shutil import move
from datetime import datetime
from contextlib import nullcontext
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# tensorboard
from torch.utils.tensorboard import SummaryWriter

import timm

from datasets.dataset_attributor import *
from utils.losses import *
from models.A import *

# for multiprocessing
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

# for removing damaged images
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def parse_args():
    parser = argparse.ArgumentParser()
    
    ## job
    parser.add_argument("--id", type=int, help="unique ID from Slurm")
    parser.add_argument("--run_name", type=str, default="freq", help="run name")

    parser.add_argument("--seed", type=int, default=3721, help="seed")

    ## multiprocessing
    parser.add_argument('--dist_backend', default='nccl', choices=['gloo', 'nccl'], help='multiprocessing backend')
    parser.add_argument('--master_addr', type=str, default="127.0.0.1", help='address')
    parser.add_argument('--master_port', type=int, default=3721, help='address')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    
    ## dataset
    parser.add_argument("--iut_paths_file", type=str, default="/dataset/iut_files.txt", help="path to the file with paths for image under test") # each line of this file should contain "/path/to/image.ext i", i is an integer represents classes
    parser.add_argument("--val_paths_file", type=str, help="path to the validation set")
    parser.add_argument("--test_paths_file", type=str, help="path to the test set")
    parser.add_argument("--n_c_samples", type=int, help="samples per classes (None for non-controlled)")
    parser.add_argument("--val_n_c_samples", type=int, help="samples per classes for validation set (None for non-controlled)")
    parser.add_argument("--test_n_c_samples", type=int, help="samples per classes for test set (None for non-controlled)")

    parser.add_argument("--n_classes_model", type=int, default=8, help="number of classes for models")
    parser.add_argument("--n_classes_sem", type=int, default=6, help="number of classes for sementics")

    parser.add_argument("--image_size", type=int, default=128, help="size of images after resize")
    parser.add_argument("--crop_size", type=int, default=128, help="size of cropped images")

    parser.add_argument("--quality", type=int, nargs=2, default=[70, 100], help="JPEG quality")

    parser.add_argument("--workers", type=int, default=0, help="number of cpu threads to use during batch generation")

    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches") # note that batch size will be multiplied by n_classes in the end

    ## model
    parser.add_argument('--load_path', type=str, help='pretrained checkpoint for continued training (A)')

    ## awl
    parser.add_argument('--load_path_awl', type=str, help='pretrained checkpoint for continued training (awl)')

    ## optimizer and scheduler
    parser.add_argument("--optim", choices=['adam', 'adamw'], default='adamw', help="optimizer")

    parser.add_argument('--factor', type=float, default=0.1, help='factor of decay')

    parser.add_argument('--patience', type=int, default=5, help='numbers of epochs to decay for ReduceLROnPlateau scheduler (None to disable)')

    parser.add_argument('--decay_epoch', type=int, help='numbers of epochs to decay for StepLR scheduler (low priority, None to disable)')

    ## training
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")

    parser.add_argument("--cond_epoch", type=int, default=0, help="epoch to start training from")
    
    parser.add_argument("--n_early", type=int, default=10, help="number of epochs for early stopping")

    parser.add_argument("--cond_state", type=str, help="state file for continued training")

    ## losses
    parser.add_argument("--weight_mode", type=str, default='manual', choices=['manual', 'awl'], help="mode for loss weights")

    parser.add_argument("--con_mode", type=str, default='supcon', choices=['supcon', 'triplet', 'none'], help="mode for contrastive learning")

    parser.add_argument("--lambda_att", type=float, default=0.2, help="att loss weight")
    parser.add_argument("--lambda_con", type=float, default=0, help="con loss weight")

    ## log
    parser.add_argument('--save_dir', type=str, default='.', help='dir to save checkpoints and logs')
    parser.add_argument("--log_interval", type=int, default=0, help="interval between saving image samples")
    
    args = parser.parse_args()

    return args

def init_env(args, local_rank, global_rank):
    # for debug only
    #torch.autograd.set_detect_anomaly(True)

    torch.cuda.set_device(local_rank)

    args, checkpoint_dir = init_env_multi(args, global_rank)

    return args, checkpoint_dir

def init_env_multi(args, global_rank):
    setup_for_distributed(global_rank == 0)

    if (args.id is None):
        args.id = datetime.now().strftime("%Y%m%d%H%M%S")

    # set random number
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # checkpoint dir
    checkpoint_dir = args.save_dir + "/checkpoints/" + str(args.id) + "_" + args.run_name
    if global_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if (args.weight_mode == 'awl'):
        args.lambda_att = 0.0
        args.lambda_con = 0.0
        args.lambda_cls = 0.0

    # finalizing args, print here
    print(args)

    return args, checkpoint_dir

def init_models(args):
    A = Attributor(args.crop_size, args.n_classes_model).cuda()

    print(A)

    return A

def init_awl(args):
    if (args.con_mode == 'none'):
        awl = AutomaticWeightedLoss(1).cuda()
    else:
        awl = AutomaticWeightedLoss(2).cuda()
    return awl

def init_dataset(args, global_rank, world_size, val = False, test = False):
    assert not(val and test) # val and test cannot be both True

    # return None if no validation set provided
    if (val and args.val_paths_file is None):
        print('No val set!')
        return None, None
    
    # return None if no test set provided
    if (test and args.test_paths_file is None):
        print('No test set!')
        return None, None
    
    # switch between train/val/test
    if (val):
        paths_file = args.val_paths_file
        n_c_samples = args.val_n_c_samples

        set_name = 'Val'

    elif (test):
        paths_file = args.test_paths_file
        n_c_samples = args.test_n_c_samples

        set_name = 'Test'

    else:
        paths_file = args.iut_paths_file
        n_c_samples = args.n_c_samples

        set_name = 'Train'

    dataset = AttributorDataset(global_rank,
                                paths_file,
                                args.id,
                                args.image_size,
                                args.crop_size,
                                args.quality,
                                n_c_samples,
                                val, test)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    
    local_batch_size = args.batch_size // world_size

    if (not val and not test):
        print('Local batch size is {} ({}//{})!'.format(local_batch_size, args.batch_size, world_size))

    dataloader = DataLoader(dataset=dataset, batch_size=local_batch_size, num_workers=args.workers, pin_memory=True, drop_last=True, sampler=sampler, collate_fn=collate_fn)

    n_drop = len(dataloader.dataset) - len(dataloader) * args.batch_size
    print('{} set size is {} (drop_last {})!'.format(set_name, len(dataloader) * args.batch_size, n_drop))

    return sampler, dataloader

def init_optims(args, world_size,
                A,
                awl):
    
    # Optimizers
    local_lr = args.lr / world_size

    print('Local learning rate is %.3e (%.3e/%d)!' % (local_lr, args.lr, world_size))

    if (args.optim == 'adam'):
        optimizer = torch.optim.Adam([
            {'params':A.parameters(), 'lr':local_lr},
            {'params': awl.parameters(), 'lr': 0.01}
        ])
    elif (args.optim == 'adamw'):
        optimizer = torch.optim.AdamW([
            {'params':A.parameters(), 'lr':local_lr},
            {'params': awl.parameters(), 'lr': 0.01}
        ])
    else:
        print("Unrecognized optimizer %s" % args.optim)
        sys.exit()

    print("Using optimizer {}".format(args.optim))

    return optimizer

def init_schedulers(args, optimizer):
    lr_scheduler = None

    # high priority for ReduceLROnPlateau (validation set required)
    if (args.val_paths_file and args.patience):
        print("Using scheduler ReduceLROnPlateau")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                    factor = args.factor,
                                                    patience = args.patience)
    # low priority StepLR
    elif (args.decay_epoch):
        print("Using scheduler StepLR")
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,
                                                    step_size = args.decay_epoch,
                                                    gamma = args.factor)
    
    else:
        print("No scheduler used")

    return lr_scheduler

def load_dicts(args,
                A,
                awl):
    # Load pretrained models
    if args.load_path != None and args.load_path != 'timm':
        print('Load pretrained model: {}'.format(args.load_path))

        A.load_state_dict(torch.load(args.load_path))

    if args.weight_mode == 'awl' and args.load_path_awl != None:
        print('Load pretrained model: {}'.format(args.load_path_awl))

        awl.load_state_dict(torch.load(args.load_path_awl))

    return A, awl

# for saving checkpoints
def save_checkpoints(args, checkpoint_dir, id, epoch, save_best, last_best, get_module,
                    A,
                    awl):
    if (get_module):
        net_A = A.module
        net_awl = awl.module
    else:
        net_A = A
        net_awl = awl.module

    # always remove save last and remove previous
    torch.save(net_A.state_dict(),
                os.path.join(checkpoint_dir, str(id) + "_last_" + str(epoch) + '.pth'))

    last_pth = os.path.join(checkpoint_dir, str(id) + "_last_" + str(epoch - 1) + '.pth')
    if (os.path.exists(last_pth)):
        os.remove(last_pth)

    # save best
    if (save_best):
        torch.save(net_A.state_dict(),
                os.path.join(checkpoint_dir, str(id) + "_best_" + str(epoch) + '.pth'))

        # last_best_pth = os.path.join(checkpoint_dir, str(id) + "_best_" + str(last_best) + '.pth')
        # if (os.path.exists(last_best_pth)):
        #     os.remove(last_best_pth)

    if args.weight_mode == 'awl':
        torch.save(net_A.state_dict(),
                os.path.join(checkpoint_dir, str(id) + "_last_" + str(epoch) + '_awl.pth'))

        last_pth = os.path.join(checkpoint_dir, str(id) + "_last_" + str(epoch - 1) + '_awl.pth')
        if (os.path.exists(last_pth)):
            os.remove(last_pth)

def predict_loss_att(args, data, A,
                criterion_CE, criterion_BCE, criterion_Con, awl):
    # load data
    in_iuts, in_m_ls, in_s_ls = data

    in_iuts = in_iuts.to('cuda', non_blocking=True)
    in_m_ls = in_m_ls.to('cuda', non_blocking=True)
    in_s_ls = in_s_ls.to('cuda', non_blocking=True)

    factor = in_iuts.shape[1]

    # combine classes into batches
    in_iuts = torch.reshape(in_iuts, (in_iuts.shape[0] * in_iuts.shape[1], in_iuts.shape[2], in_iuts.shape[3], in_iuts.shape[4]))
    in_m_ls = torch.reshape(in_m_ls, (1, in_m_ls.shape[0] * in_m_ls.shape[1])).squeeze(0)
    in_s_ls = torch.reshape(in_s_ls, (1, in_s_ls.shape[0] * in_s_ls.shape[1])).squeeze(0)

    # predict
    out_att = A(in_iuts)
    
    # calculate loss
    loss_att = criterion_CE(out_att, in_m_ls)
    if (criterion_Con):
        fea = A.module.forward_features(in_iuts)
        loss_con = criterion_Con(fea, in_m_ls)
    else:
        loss_con = 0.0

    if (criterion_Con):
        if args.weight_mode == 'awl':
            loss = awl(loss_att, loss_con) / factor
        else:
            loss = (args.lambda_att * loss_att + args.lambda_con * loss_con) / factor
    else:
        if args.weight_mode == 'awl':
            loss = awl(loss_att) / factor
        else:
            loss = (args.lambda_att * loss_att) / factor

    # culculate accuracy
    preds_att = torch.argmax(F.log_softmax(out_att, dim = 1), dim = 1)
    n_correct_att = (preds_att == in_m_ls).sum().item() / factor

    return loss, loss_att, loss_con, \
           n_correct_att

def predict_loss_att_full(args, data, A,
                criterion_CE, criterion_BCE, criterion_Con, awl):
    # load data
    in_iuts, in_m_ls, _ = data

    in_iuts = in_iuts.to('cuda', non_blocking=True)
    in_m_ls = in_m_ls.to('cuda', non_blocking=True)

    factor = in_iuts.shape[1]

    # combine classes into batches
    in_iuts = torch.reshape(in_iuts, (in_iuts.shape[0] * in_iuts.shape[1], in_iuts.shape[2], in_iuts.shape[3], in_iuts.shape[4]))
    in_m_ls = torch.reshape(in_m_ls, (1, in_m_ls.shape[0] * in_m_ls.shape[1])).squeeze(0)

    B, C, H, W = in_iuts.shape
    reshaped = torch.reshape(in_iuts, (B, C // 3, 3, H, W))
    permuted = torch.permute(reshaped, (1, 0, 2, 3, 4))

    out_att_sum = None

    loss_con_sum = None

    for p in permuted:
        # predict
        out_att = A(p)

        if (out_att_sum is None):
            out_att_sum = out_att
        else:
            out_att_sum += out_att
        
        # calculate con loss
        if (criterion_Con):
            fea = A.module.forward_features(p)
            loss_con = criterion_Con(fea, in_m_ls)
        else:
            loss_con = 0.0

        if (loss_con_sum is None):
            loss_con_sum = loss_con
        else:
            loss_con_sum += loss_con

    # calculate att output
    out_att_avg = out_att_sum / (C // 3)

    # calculate final loss
    loss_att = criterion_CE(out_att_avg, in_m_ls) # no need to divide, out_att_avg is for one mean sample
    loss_con_avg = loss_con_sum / (C // 3)

    if (criterion_Con):
        if args.weight_mode == 'awl':
            loss = awl(loss_att, loss_con_avg) / factor
        else:
            loss = (args.lambda_att * loss_att + args.lambda_con * loss_con_avg) / factor
    else:
        if args.weight_mode == 'awl':
            loss = awl(loss_att) / factor
        else:
            loss = (args.lambda_att * loss_att) / factor

    # culculate accuracy
    preds_att = torch.argmax(F.log_softmax(out_att_avg, dim = 1), dim = 1)
    n_correct_att = (preds_att == in_m_ls).sum().item() / factor

    return loss, loss_att, loss_con_avg, \
           n_correct_att

def reset_optim_lr(optimizer, lr, world_size):
    local_lr = lr / world_size

    for g in optimizer.param_groups:
        g['lr'] = local_lr

    return optimizer

def init_early_stopping():
    best_val_loss = float('inf')
    n_last_epochs = 0
    early_stopping = False

    return best_val_loss, n_last_epochs, early_stopping

def save_state(checkpoint_dir, id, epoch, last_best,
                optimizer, lr_scheduler,
                best_val_loss):
    
    state = {'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(),
             'best_val_loss': best_val_loss}

    save_best_pth = os.path.join(checkpoint_dir, str(id) + "_best_" + str(epoch) + '_state.pth')
    torch.save(state, save_best_pth)

    last_best_pth = os.path.join(checkpoint_dir, str(id) + "_best_" + str(last_best) + '_state.pth')
    if (os.path.exists(last_best_pth)):
        os.remove(last_best_pth)

    print('State saved to %s with best val loss %.3e.' % (save_best_pth, best_val_loss))

def train(args, global_rank, world_size, sync, get_module,
            checkpoint_dir,
            A,
            train_sampler, dataloader, val_sampler, val_dataloader, test_sampler, test_dataloader,
            optimizer,
            lr_scheduler,
            awl,
            prev_best_val_loss):
    # Losses that are built-in in PyTorch
    criterion_CE = nn.CrossEntropyLoss().cuda() # ref: https://pytorch.org/docs/1.9.1/generated/torch.nn.CrossEntropyLoss.html
    criterion_BCE = nn.BCEWithLogitsLoss().cuda()

    if (args.con_mode == 'supcon'):
        criterion_Con = SupConLoss().cuda()
    elif (args.con_mode == 'triplet'):
        criterion_Con = TripletLoss().cuda()
    elif (args.con_mode == 'none'):
        criterion_Con = None
    else:
        print("Unsupported contrastive loss %s" % args.con_mode)
        sys.exit()

    print("Using contrastive loss {}".format(args.con_mode))

    # tensorboard
    if global_rank == 0:
        os.makedirs(args.save_dir + "/logs/", exist_ok=True)
        writer = SummaryWriter(args.save_dir + "/logs/" + str(args.id) + "_" + args.run_name)

    # for early stopping
    best_val_loss, n_last_epochs, early_stopping = init_early_stopping()

    last_best = -1
    new_best = -1

    if (prev_best_val_loss is not None):
        best_val_loss = prev_best_val_loss

        print("Best val loss set to %.3e" % (prev_best_val_loss))

    # for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = args.cond_epoch
    for epoch in range(start_epoch, args.n_epochs):

        train_sampler.set_epoch(epoch)
        
        print('Starting Epoch {}'.format(epoch))

        # loss sum for epoch
        epoch_loss = 0
        epoch_loss_att = 0
        epoch_loss_con = 0

        epoch_val_loss = 0

        # for accuracy
        train_n_correct_att = 0
        
        # ------------------
        #  Train step
        # ------------------
        with A.join() if get_module else nullcontext(), awl.join() if get_module else nullcontext(): # get_module indicates using DDP         
            for step, data in enumerate(dataloader):
                curr_steps = epoch * len(dataloader) + step

                A.train()
                awl.train()

                if (sync): optimizer.synchronize()
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    loss, loss_att, loss_con, n_correct_att = predict_loss_att(args, data, A, criterion_CE, criterion_BCE, criterion_Con, awl)

                # backward prop
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # log losses for epoch
                epoch_loss += loss
                epoch_loss_att += loss_att
                epoch_loss_con += loss_con

                # log accuracy
                train_n_correct_att += n_correct_att

                # --------------
                #  Log Progress (for certain steps)
                # --------------
                if args.log_interval != 0 and step % args.log_interval == 0 and global_rank == 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
                            f"[Epoch {epoch}, Batch {step}/{len(dataloader) - 1}]"
                            f"[Loss {loss:.3e}]"
                            f"[Loss Att {loss_att:.3e}]"
                            f"[Loss Con {loss_con:.3e}]")

                    writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], curr_steps)

                    writer.add_scalar("Loss/Train", loss, curr_steps)
                    writer.add_scalar("Loss/Att", loss_att, curr_steps)
                    writer.add_scalar("Loss/Con", loss_con, curr_steps)

        # ------------------
        #  Validation
        # ------------------
        if (val_sampler and val_dataloader):

            val_sampler.set_epoch(epoch)

            # for accuracy
            val_n_correct = 0

            A.eval()
            awl.eval()

            for step, data in enumerate(val_dataloader):
                with torch.no_grad():
                    loss, _, _, n_correct_att = predict_loss_att_full(args, data, A, criterion_CE, criterion_BCE, criterion_Con, awl)

                    epoch_val_loss += loss.item()

                    val_n_correct += n_correct_att

            # early 
            if epoch_val_loss <= best_val_loss:
                best_val_loss = epoch_val_loss
                n_last_epochs = 0

                new_best = epoch
            else:
                n_last_epochs += 1

                if (n_last_epochs > args.n_early):
                    early_stopping = True
        else:
            new_best = epoch

        # ------------------
        #  Test
        # ------------------
        if (test_sampler and test_dataloader):

            test_sampler.set_epoch(epoch)

            # for accuracy
            test_n_correct = 0

            A.eval()
            awl.eval()

            for step, data in enumerate(test_dataloader):
                with torch.no_grad():
                    _, _, _, n_correct_att = predict_loss_att_full(args, data, A, criterion_CE, criterion_BCE, criterion_Con, awl)

                    test_n_correct += n_correct_att

        # ------------------
        #  Step
        # ------------------
        lr_before_step = optimizer.param_groups[0]['lr']

        if (lr_scheduler):
            if (args.val_paths_file and args.patience):
                lr_scheduler.step(epoch_val_loss) # ReduceLROnPlateau
            elif (args.decay_epoch):
                lr_scheduler.step() # StepLR
            else:
                print("Error in scheduler step")
                sys.exit()

        # --------------
        #  Log Progress (for epoch)
        # --------------
        # loss average for epoch
        if (global_rank == 0):
            epoch_loss_avg = epoch_loss / len(dataloader)
            epoch_loss_att_avg = epoch_loss_att / len(dataloader)
            epoch_loss_con_avg = epoch_loss_con / len(dataloader)

            # accuracy
            local_batch_size = args.batch_size // world_size

            train_acc = train_n_correct_att / local_batch_size / len(dataloader)

            if (val_dataloader):
                epoch_val_loss_avg = epoch_val_loss / len(val_dataloader)
                best_val_loss_avg = best_val_loss / len(val_dataloader)

                val_acc = val_n_correct / local_batch_size / len(val_dataloader)
            else:
                epoch_val_loss_avg = 0
                best_val_loss_avg = 0

                val_acc = 0

            if (test_dataloader):
                test_acc = test_n_correct / local_batch_size / len(test_dataloader)
            else:
                test_acc = 0

            # global lr (use before-step lr)
            global_lr = lr_before_step * world_size

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
                  f"[Epoch {epoch}/{args.n_epochs - 1}]"
                  f"[Loss {epoch_loss_avg:.3e}]"
                  f"[Loss Att {epoch_loss_att_avg:.3e}]"
                  f"[Loss Con {epoch_loss_con_avg:.3e}]"
                  f"[Val Loss {epoch_val_loss_avg:.3e} (Best {best_val_loss_avg:.3e} @{n_last_epochs:d})]"
                  f"[Acc {train_acc:.3f}]"
                  f"[Val Acc {val_acc:.3f}]"
                  f"[Test Acc {test_acc:.3f}]"
                  f"[LR {global_lr:.3e}]")

            writer.add_scalar("Epoch LearningRate", global_lr, epoch)

            writer.add_scalar("Epoch Loss/Train", epoch_loss_avg, epoch)
            writer.add_scalar("Epoch Loss/Att", epoch_loss_att_avg, epoch)
            writer.add_scalar("Epoch Loss/Con", epoch_loss_con_avg, epoch)

            writer.add_scalar("Epoch Acc/Train", train_acc, epoch)

            writer.add_scalar("Epoch Loss/Val", epoch_val_loss_avg, epoch)
            writer.add_scalar("Epoch Acc/Val", val_acc, epoch)

            writer.add_scalar("Epoch Acc/Test", test_acc, epoch)

            # save model parameters
            if global_rank == 0:
                save_checkpoints(args, checkpoint_dir, args.id, epoch,
                                 new_best == epoch, last_best,
                                 get_module,
                                 A, awl)

            # save lr and best val loss
            if global_rank == 0 and new_best == epoch:
                save_state(checkpoint_dir, args.id, epoch, last_best,
                 optimizer, lr_scheduler,
                 best_val_loss)

            # update last best to new best
            if global_rank == 0 and new_best == epoch:
                last_best = new_best
                    
        # reset early stopping when learning rate changed
        lr_after_step = optimizer.param_groups[0]['lr']
        if (lr_after_step != lr_before_step):
            print("LR changed to %.3e" % (lr_after_step * world_size))

            best_val_loss, n_last_epochs, early_stopping = init_early_stopping()

        # check early_stopping
        if (early_stopping):
            print('Early stopping')
            break

    print('Finished training')

    if global_rank == 0:
        writer.close()

    pass