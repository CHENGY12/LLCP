import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import pickle
from termcolor import colored
from sklearn import preprocessing
from DataLoader import VideoQADataLoadertest
import time

from models_cvae import VAE
from config import cfg, cfg_from_file
import ipdb


def validate(cfg, model, data, device):
    model.eval()
    cos1 = nn.CosineSimilarity(dim=1, eps=1e-6)
    le = preprocessing.LabelEncoder()
    le.fit(['A','C','F', 'I' ,'R','U'])
    accdic = {
        'acc': [0,0], 
        'Aacc': [0,0],
        'Racc': [0,0],
        'Cacc': [0,0],
        'Iacc': [0,0],
            }
    all_preds = []
    all_answers = []
    print('validating...')

    with torch.no_grad():
        for batch in tqdm(iter(data), total=len(data)):
            obj_feat, app_feat, quest, ans, cand_ans, qt, y_idx, n_idx = batch
            ans = ans.to(device)
            obj_feat = obj_feat.contiguous().view(obj_feat.size(0)*obj_feat.size(1),obj_feat.size(2),obj_feat.size(3))
            obj_feat = obj_feat.to(device) # shape T 5 1024
            obj_feat_current = obj_feat[:,0,:] # shape T 1 1024
            obj_feat_pre = obj_feat[:,1:,:] # shape T 4 1024
            # T = obj_feat_current.shape[0]
            cand_ans = cand_ans.to(device)
            quest = quest.to(device)


#####################################################################
            z = torch.zeros(obj_feat_pre.size(0),10).cuda()
            x_reconst = model.inference(z,obj_feat_pre) # shape [T 1024]
            x_residel = obj_feat_current - x_reconst # shape [T 1024]
            dis = torch.linalg.norm(x_residel, dim=1, ord=2)
            x_index = torch.argmax(dis.clone().detach())
            obj_feat_test = obj_feat_current[x_index]
            obj_feat_test = obj_feat_test.unsqueeze(0)
#####################################################################

            app_feat = app_feat.to(device)
            app_feat = app_feat.reshape(app_feat.shape[0]*app_feat.shape[1]*app_feat.shape[2],app_feat.shape[3])
            app_feat_test = torch.mean(app_feat, dim=0)
            app_feat_test = torch.unsqueeze(app_feat_test,0)

            qt = [*le.inverse_transform(qt.squeeze(0))][0]
            if qt in ['A']:
                simi = cos1(cand_ans.squeeze(),obj_feat_test+cfg.test.rate*app_feat_test) 
                predicted = simi.argmax() 
            elif qt in ['R']:
                simi = cos1(cand_ans.squeeze(),obj_feat_test) 
                predicted = simi.argmax() 
            elif qt in ['C'] :
                simi = cos1(quest,obj_feat_test) 
                if simi.item() > cfg.test.thr:
                    indices = y_idx
                    if len(indices)>=1:
                        indices_list = torch.cat([*indices]).tolist()
                        if len(indices_list)==1:
                            predicted = torch.tensor(indices_list)
                        else:
                            simi = cos1(cand_ans.squeeze(),obj_feat_test)
                            simi[indices_list]+=1
                            predicted = simi.argmax()
                    else:
                        continue
                else:
                    indices = n_idx
                    if len(indices)>=1:
                        indices_list = torch.cat([*indices]).tolist()                               
                        if len(indices_list)==1:
                            predicted = torch.tensor(indices_list)
                        else:
                            simi = cos1(cand_ans.squeeze(),obj_feat_test)
                            simi[indices_list]+=1
                            predicted = simi.argmax()
                    else:
                        continue

            elif qt in ['I'] :
                simi = cos1(quest,obj_feat_test)
                if simi.item() > cfg.test.thr:
                    indices = y_idx
                    if len(indices)>=1:
                        indices_list = torch.cat([*indices]).tolist()
                        if len(indices_list)==1:
                            predicted = torch.tensor(indices_list)
                        else:
                            simi = cos1(cand_ans.squeeze(),obj_feat_test) 
                            simi[indices_list]+=1
                            predicted = simi.argmax()
                    else:
                        continue
                else:
                    indices = n_idx
                    if len(indices)>=1:
                        indices_list = torch.cat([*indices]).tolist()
                        if len(indices_list)==1:
                            predicted = torch.tensor(indices_list)
                        else:
                            simi = cos1(cand_ans.squeeze(),obj_feat_test)
                            simi[indices_list]+=1
                            predicted = simi.argmax()
                    else:
                        continue
            else:
                continue
            predicted = predicted.to(device)   
            #True answer and accuracy calcualtions
            all_preds.append(predicted)
            all_answers.append(ans.squeeze())
            accdic['acc'][0] = accdic['acc'][0]+ (predicted ==ans).cpu().numpy()
            accdic['acc'][1] = accdic['acc'][1]+ 1
            accdic[qt + 'acc'][0] = accdic[qt +'acc'][0]+ (predicted ==ans).cpu().numpy()
            accdic[qt +'acc'][1] = accdic[qt +'acc'][1]+ 1

    acc = accdic['acc'][0]/ accdic['acc'][1]
    Aacc = accdic['Aacc'][0]/ accdic['Aacc'][1]
    Racc = accdic['Racc'][0]/ accdic['Racc'][1]
    Cacc = accdic['Cacc'][0]/ accdic['Cacc'][1]
    Iacc = accdic['Iacc'][0]/ accdic['Iacc'][1]


    return accdic, acc, Aacc, Cacc, Iacc, Racc




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/sutd-traffic_transition.yml', type=str)
    parser.add_argument('--exp_name', type=str, default='llcp', help='specify experiment name')
    parser.add_argument('--layer_size', type=int, default=256, help='specify layer size')
    parser.add_argument('--latent_size', type=int, default=10, help='specify latent size')
    parser.add_argument('--colatent_size', type=int, default=16, help='specify condition latent size')              
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, args.exp_name+'_layer_size'+str(args.layer_size)+'_latent_size'+str(args.latent_size)+'_colatent_size'+str(args.colatent_size))
    ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model_cvae49.pt')
    assert os.path.exists(ckpt)
    loaded = torch.load(ckpt, map_location='cpu')

    latent_layer_size = args.layer_size
    latent_size = args.latent_size
    con_latent_size = args.colatent_size

    model = VAE(
        encoder_layer_sizes=[1024, latent_layer_size],
        latent_size=latent_size,
        decoder_layer_sizes=[latent_layer_size, 1024],
        con_encoder_layer_sizes=[1024,latent_layer_size], 
        con_latent_size=con_latent_size
        ).to(device)
    
    model.load_state_dict(loaded['state_dict'])


    test_loader_kwargs = {
        'question_pt': cfg.dataset.test_question_pt,
        'appearance_feat': cfg.dataset.appearance_feat,
        'object_feat': cfg.dataset.object_feat,
        'batch_size': 1,
        'num_workers': cfg.num_workers,
        'shuffle': False
    }
    test_loader = VideoQADataLoadertest(**test_loader_kwargs)


    accdic, acc, accA, accC, accI, accR = validate(cfg, model, test_loader,device)
    count = accdic['acc'][1]
    sys.stdout.write('~~~~~~ Test Accuracy: {test_acc}, counts:{count} ~~~~~~~\n'.format(
        test_acc=colored("{:.4f}".format(acc.item()), "red", attrs=['bold']),
        count=colored("{:.4f}".format(count), "red", attrs=['bold'])))
    sys.stdout.flush()
    count = accdic['Aacc'][1]
    sys.stdout.write('~~~~~~ Attribution Test Accuracy: {test_acc}, counts:{count} ~~~~~~~\n'.format(
        test_acc=colored("{:.4f}".format(accA.item()), "red", attrs=['bold']),
        count=colored("{:.4f}".format(count), "red", attrs=['bold'])))
    sys.stdout.flush()
    count = accdic['Cacc'][1]
    sys.stdout.write('~~~~~~ Counterfactual Test Accuracy: {test_acc}, counts:{count} ~~~~~~~\n'.format(
        test_acc=colored("{:.4f}".format(accC.item()), "red", attrs=['bold']),
        count=colored("{:.4f}".format(count), "red", attrs=['bold'])))
    sys.stdout.flush()
    count = accdic['Iacc'][1]
    sys.stdout.write('~~~~~~ Introspection Test Accuracy: {test_acc}, counts:{count} ~~~~~~~\n'.format(
        test_acc=colored("{:.4f}".format(accI.item()), "red", attrs=['bold']),
        count=colored("{:.4f}".format(count), "red", attrs=['bold'])))
    sys.stdout.flush()
    count = accdic['Racc'][1]
    sys.stdout.write('~~~~~~ Reverse Reasoning Test Accuracy: {test_acc}, counts:{count} ~~~~~~~\n'.format(
        test_acc=colored("{:.4f}".format(accR.item()), "red", attrs=['bold']),
        count=colored("{:.4f}".format(count), "red", attrs=['bold'])))
    sys.stdout.flush()

    save_dic = {
        'acc': [acc,accdic['acc'][0],accdic['acc'][1]],
        'accA': [accA,accdic['Aacc'][0],accdic['Aacc'][1]],
        'accC': [accC,accdic['Cacc'][0],accdic['Cacc'][1]],
        'accR': [accR,accdic['Racc'][0],accdic['Racc'][1]],
        'accI': [accI,accdic['Iacc'][0],accdic['Iacc'][1]],
    }
    timestr = time.strftime("%Y%m%d-%H%M%S")
    path_log = 'accuracies'+timestr
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    path = os.path.join('accuracies'+timestr,f'_accdictionary.pkl')
    with open(path , 'wb') as f:
        pickle.dump(save_dic, f)
        

