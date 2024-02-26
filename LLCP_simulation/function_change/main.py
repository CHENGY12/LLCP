import os, sys

import torch
from sklearn.metrics import recall_score, f1_score, precision_score
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import logging
from termcolor import colored
from torch.utils.data import DataLoader
from models_cvae import VAE
# from models_cvae_good import VAE


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

# from utils import SimulatedTrainDataset, SimulatedTestDataset, ForeverDataIterator
from utils import SimulatedTrainDataset, ForeverDataIterator, parser_data


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def loss_fn(recon_x, x, mean, log_var):
        # BCE = torch.nn.functional.binary_cross_entropy(
        #     recon_x, x, reduction='sum')
        BCE = torch.sum(torch.sqrt((recon_x - x) ** 2))
        # print(BCE)
        # exit()
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)
        # return (BCE) / x.size(0)

    train_dataset = SimulatedTrainDataset(args.train_data_path)
    # print(len(train_dataset))
    # exit()
    data_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=1,
                             num_workers=args.num_workers)
    train_iterator = ForeverDataIterator(data_loader=data_loader)

    test_dataset = SimulatedTrainDataset(args.test_data_path)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=1)


    model = VAE(
        encoder_layer_sizes=[1, 10],
        latent_size=1,
        decoder_layer_sizes=[5, 1],
        con_encoder_layer_sizes=[1, 10],
        # input content: 2 dims of neighbour, 1 dims of previous, 1 dims of env
        con_latent_size=3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-6)

    for step in range(args.total_steps):
        model.train()

        batch_x_list = []
        batch_c_list = []
        batch_y_list = []
        for batch_vedio_id in range(args.batch_size):
            one_vedion_data = next(train_iterator)
            # print(len(one_vedion_data))
            # exit()
            one_vedio_x, one_vedio_c, one_vedio_y = parser_data(one_vedion_data)
            batch_x_list.append(one_vedio_x)
            batch_c_list.append(one_vedio_c)
            batch_y_list.append(one_vedio_y)

        batch_current_x = torch.vstack(batch_x_list).float().cuda()
        batch_previous_c = torch.unsqueeze(torch.vstack(batch_c_list), dim=-1).float().cuda()
        batch_current_y = torch.vstack(batch_y_list).float().cuda()

        recon_x, mean, log_var, z = model(batch_current_x, batch_previous_c)
        # print(z.shape)
        # exit()
        loss = loss_fn(recon_x, batch_current_x, mean, log_var)
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)  # need to be tested
        optimizer.step()
        # exit()

        if (step + 1) % (75 * 1) == 0:
            print(step,  loss)
            # continue
            count = 0
            positive_rec_loss_list = []
            negative_rec_loss_list = []
            model.eval()

            true_positive_count = 0
            false_positive_count = 0
            true_negative_count = 0
            false_negative_count = 0

            # threshold = 0.2830
            threshold = 0.3430
            correct_rec_resi_list = []
            ground_truth_list = []
            predict_list = []

            label_0_error = 0.0
            for one_vedion_data in test_data_loader:
                one_vedio_x, one_vedio_c, one_vedio_y = parser_data(one_vedion_data)

                # print(one_vedio_x.shape)
                # print(one_vedio_c.shape)
                # print(one_vedio_y.shape)
                obj_feat_current = one_vedio_x.float().cuda()
                obj_feat_pre = torch.unsqueeze(one_vedio_c, dim=-1).float().cuda()
                x_residels = []
                z_samples = []
                # for sample in range(100):
                z = torch.zeros(obj_feat_current.shape)
                z = z.cuda()
                x_reconst = model.inference(z, obj_feat_pre)  # shape [T 1024]
                x_residel = torch.abs(obj_feat_current - x_reconst)  # shape [T 1024]

                x_residels.append(x_residel)
                z_samples.append(z)

                if torch.max(one_vedio_y) == 1:
                    ground_truth_list.append(np.asarray(1, dtype=int))

                    if torch.argmax(x_residel) == torch.argmax(one_vedio_y):
                        predict_list.append(np.asarray(1, dtype=int))
                        true_positive_count += 1
                    else:
                        predict_list.append(np.asarray(0, dtype=int))
                        false_negative_count += 1
                else:
                    ground_truth_list.append(np.asarray(0))
                    no_addcident_vedio_max_rec_err = torch.max(x_residel)
                    label_0_error += no_addcident_vedio_max_rec_err
                    if no_addcident_vedio_max_rec_err > threshold:
                        false_positive_count += 1
                        predict_list.append(np.asarray(1))
                    else:
                        true_negative_count += 1
                        predict_list.append(np.asarray(0))

                if torch.argmax(x_residel) == torch.argmax(one_vedio_y):
                    if torch.max(one_vedio_y) == 1:
                        positive_rec_loss_list.append([torch.argmax(x_residel)])
                    else:
                        negative_rec_loss_list.append([torch.argmax(x_residel)])
                    count += 1
                    pass

            print(true_positive_count)
            print(false_negative_count)
            print("label_o_mean_error:", label_0_error / 618)
            # print("recall:", true_positive_count / (true_positive_count + false_negative_count))
            # print(ground_truth_list)
            # print(predict_list)
            # exit()
            # print("recall2: ", recall_score(y_true=ground_truth_list, y_pred=predict_list, average="marco"))
            print("recall2: ", recall_score(y_true=ground_truth_list, y_pred=predict_list))
            # print("f1 score:", f1_score(y_true=ground_truth_list, y_pred=predict_list, average="marco"))
            print("f1 score:", f1_score(y_true=ground_truth_list, y_pred=predict_list))
            print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=3, type=int)
    # parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--lr", default=0.003, type=float)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--total_steps", default=500, type=int)
    parser.add_argument("--train_data_path", default="dataset/train.pkl", type=str)
    parser.add_argument("--test_data_path", default="dataset/test.pkl", type=str)
    parser.add_argument("--num_workers", default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)
    pass

