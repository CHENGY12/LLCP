import os, sys

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import logging
from termcolor import colored
from models_cvae import VAE
from DataLoader import VideoQADataLoader
from config import cfg, cfg_from_file
import ipdb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()


def todevice(tensor, device):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)


def train(cfg):
    logging.info("Create train_loader.........")
    train_loader_kwargs = {
        'question_pt': cfg.dataset.train_question_pt,
        'appearance_feat': cfg.dataset.appearance_feat,
        'object_feat': cfg.dataset.train_object_feat,
        'video_list': cfg.dataset.video_list,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': True
    }
    train_loader = VideoQADataLoader(**train_loader_kwargs)

    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VAE(
        encoder_layer_sizes=[1024, cfg.model.latent_layer_size],
        latent_size=cfg.model.latent_size,
        decoder_layer_sizes=[cfg.model.latent_layer_size, 1024],
        con_encoder_layer_sizes=[1024,cfg.model.latent_layer_size], 
        con_latent_size=cfg.model.con_latent_size
        ).to(device)    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    logging.info(model)

    start_epoch = 0
    logging.info("Start training........")
    for epoch in range(start_epoch, cfg.train.max_epochs):
        logging.info('>>>>>> epoch {epoch} <<<<<<'.format(epoch=colored("{}".format(epoch), "green", attrs=["bold"])))
        model.train()
        total_loss = 0
        for i, batch in enumerate(iter(train_loader)):
            progress = epoch + i / len(train_loader)
            # ipdb.set_trace()
            feat, _, _ =[todevice(x, device) for x in batch]
            feat = feat.contiguous().view(feat.shape[0]*feat.shape[1],feat.shape[2],feat.shape[3])
            feat_current_obj = feat[:,0,:]
            feat_pre_objs = feat[:,1:,:]
          
            recon_x, mean, log_var, z = model(feat_current_obj, feat_pre_objs)
            loss = loss_fn(recon_x, feat_current_obj, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=12) 
            optimizer.step()
            total_loss += loss.detach()
            avg_loss = total_loss / (i + 1)

            sys.stdout.write(
                "\rProgress = {progress}   loss = {loss}   avg_loss = {avg_loss}    exp: {exp_name}".format(
                    progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                    loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                    avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                    exp_name=cfg.exp_name))
            sys.stdout.flush()
        sys.stdout.write("\n")
        if (epoch + 1) % 10 == 0:
            optimizer = step_decay(cfg, optimizer)
        sys.stdout.flush()
        logging.info("Epoch = %s   avg_loss = %.3f " % (epoch, avg_loss))

        if (epoch+1)%1==0:
            ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            else:
                assert os.path.isdir(ckpt_dir)
            save_checkpoint(epoch, model, optimizer, os.path.join( ckpt_dir, 'model_cvae{}.pt'.format(epoch) ) )
            sys.stdout.write('\n >>>>>> save to %s <<<<<< \n' % (ckpt_dir))
            sys.stdout.flush()


def step_decay(cfg, optimizer):
    # compute the new learning rate based on decay rate
    cfg.train.lr *= 0.5
    logging.info("Reduced learning rate to {}".format(cfg.train.lr))
    sys.stdout.flush()
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.train.lr

    return optimizer


def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    time.sleep(1)
    torch.save(state, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/sutd-traffic_transition.yml', type=str)
    parser.add_argument('--exp_name', type=str, default='llcp', help='specify experiment name')
    parser.add_argument('--layer_size', type=int, default=256, help='specify layer size')
    parser.add_argument('--latent_size', type=int, default=10, help='specify latent size')
    parser.add_argument('--colatent_size', type=int, default=16, help='specify condition latent size')     
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, args.exp_name)
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, args.exp_name+'_layer_size'+str(args.layer_size)+'_latent_size'+str(args.latent_size)+'_colatent_size'+str(args.colatent_size))
    if not os.path.exists(cfg.dataset.save_dir):
        os.makedirs(cfg.dataset.save_dir)
    else:
        assert os.path.isdir(cfg.dataset.save_dir)
    log_file = os.path.join(cfg.dataset.save_dir, "log")
    if not cfg.train.restore and not os.path.exists(log_file):
        os.mkdir(log_file)
    else:
        assert os.path.isdir(log_file)

    cfg.model.latent_layer_size = args.layer_size
    cfg.model.latent_size = args.latent_size
    cfg.model.con_latent_size = args.colatent_size

    fileHandler = logging.FileHandler(os.path.join(log_file, 'stdout.log'), 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(cfg).items():
        logging.info(k + ':' + str(v))
  
    train(cfg)


if __name__ == '__main__':
    main()
