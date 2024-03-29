import os
from io import StringIO
import numpy as np
import json
import pickle
import torch
import random
import math
import h5py
from torch.utils.data import Dataset, DataLoader
import ipdb
from sklearn import preprocessing




def dist_bbox(bbox1, bbox2):
    center_1x= (bbox1[0]+bbox1[2])/2
    center_1y= (bbox1[1]+bbox1[3])/2
    center_2x= (bbox2[:,0]+bbox2[:,2])/2
    center_2y= (bbox2[:,1]+bbox2[:,3])/2
    return np.sqrt((center_1x-center_2x)**2 + (center_1y-center_2y)**2)


def invert_dict(d):
    return {v: k for k, v in d.items()}

def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab




class VideoQADatasetTrain(Dataset):

    def __init__(self, feat, index_list, nav_feat_idx,batch_size):        
        self.feat = feat
        self.index_list = index_list
        self.nav_feat_idx = nav_feat_idx
        self.batch_size = batch_size

    def __getitem__(self, index):
        nav_feat_idx = self.nav_feat_idx[index]
        nav_feat = self.feat[index]
        random_select = random.sample(range(len(nav_feat)), self.batch_size)
        nav_feat_np =[]
        for i in random_select: nav_feat_np.append(nav_feat[i])
        nav_feat_np= np.asarray(nav_feat_np)
        index_list = np.array(self.index_list[index])[random_select]
        return (
            nav_feat_np, index_list, nav_feat_idx
        )

    def __len__(self):
        return len(self.feat)




class VideoQADataset(Dataset):

    def __init__(self, answers, questions,ans_candidates,q_types,
                yes_indices, no_indices, video_ids,feat, index_list, nav_feat_idx, 
                app_feature_h5, app_video_ids,q_ids,app_feat_id_to_index, obj_feat_id_to_index):

        self.all_answers = answers
        self.all_questions = questions
        self.all_q_ids = q_ids
        self.all_ans_candidates = ans_candidates
        self.all_video_ids = video_ids

        self.q_types = q_types
        self.yes_indices = yes_indices
        self.no_indices = no_indices
        self.app_feature_h5 = app_feature_h5
        self.app_video_ids = app_video_ids
        self.feat = feat
        self.app_feat_id_to_index = app_feat_id_to_index
        self.obj_feat_id_to_index = obj_feat_id_to_index
        self.index_list = index_list
        self.nav_feat_idx = nav_feat_idx
        self.le = preprocessing.LabelEncoder()
        self.le.fit(['A','C','F', 'I' ,'R','U'])




    def __getitem__(self, index):

        ##### language #####
        ans = self.all_answers[index] if self.all_answers is not None else None
        cand_ans = self.all_ans_candidates[index]
        quest = self.all_questions[index]
        # question_idx = self.all_q_ids[index]
        qt = torch.tensor(self.le.transform([self.q_types[index]]))
        y_idx = [*self.yes_indices[index]]
        n_idx = [*self.no_indices[index]]

        ##### video #####
        video_idx = self.all_video_ids[index].item()
        if not str(video_idx) in self.obj_feat_id_to_index:
            return 
        app_index = self.app_feat_id_to_index[str(video_idx)]

        with h5py.File(self.app_feature_h5, 'r') as f_app:
            app_feat = f_app['appearance_features'][app_index]  # (8, 16, d)

        ##### object #####
        obj_index = self.obj_feat_id_to_index[str(video_idx)] # find the obj feat index by the video id
        obj_feat = torch.tensor(self.feat[obj_index])  # obtain the obj feat with obj feat index
  

        return (
            obj_feat, app_feat, quest, ans, cand_ans, qt, y_idx, n_idx
        )

    def __len__(self):
        return len(self.all_questions)


class VideoQADataLoadertest(DataLoader):

    def __init__(self, **kwargs):
        
        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        
        with open(question_pt_path, 'rb') as f: 
            obj = pickle.load(f)
            
            q_ids = obj['test_qids']
            questions = obj['questions'][q_ids]
            video_ids = obj['video_ids'][q_ids]
            answers = obj['answers'][q_ids]
 
            q_types = obj['q_type']
            q_types = q_types[q_ids]
            yes_indices = obj['yes_indices'][q_ids]
            no_indices = obj['no_indices'][q_ids]
            ans_candidates = obj['ans_candidates'][q_ids]

        print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file: # get the glocal features 
            app_video_ids = app_features_file['ids'][()] # 10080 for test
            app_features = app_features_file['appearance_features'][()] # 10080 8 16 d for test

        feat = []
        index_list = []
        nav_feat_idx = []
        object_feat = kwargs.pop('object_feat')
        print('loading object features from %s, it will cost some minutes' % (object_feat))
        counter = 0
        for video in os.listdir(object_feat):
            if int(video) in video_ids:
                with h5py.File(os.path.join(object_feat, video, 'feat_obj.h5'), 'r') as feat_file:
                    bbox = feat_file['bbox'][()]
                    feat_obj = feat_file['feat_obj'][()]
                    frame_id = feat_file['frame_id'][()] #[0,0,1,1,2,2,10,10,31,31] frame_id_obj = frame_id[index]
                    object_id = feat_file['object_id'][()]
                    # score = feat_file['score'][()]

                    feat_t = []
                    index_t = []

                    for t in range(1, frame_id.max()+1):
                        t_idx = np.nonzero(frame_id == t)[0] # index of the obj featues in this frame
                        t_idx_ = np.nonzero(frame_id == t-1)[0] # index of the obj featues in last frame
                        # feat_agent = []
                        for index in t_idx: # get each obj
                            j =object_id[index]
                            feat_agent_ = []
                            if j in object_id[t_idx_]:
                                feat_curr = feat_obj[index]
                                feat_agent_.append(feat_curr)

                                
                                index_obj = t_idx_[np.where(object_id[t_idx_] == j)[0][0]]
                                # import pdb; pdb.set_trace()
                                feat_his = feat_obj[index_obj]
                                feat_agent_.append(feat_his)

                                index_obj_all = t_idx_[np.where(t_idx_ != index_obj)[0]]
                                dist = dist_bbox(bbox[index],bbox[index_obj_all])

                                idx_dist = dist.argsort()
                                if idx_dist.size >= 2:
                                    feat_neibor1 = feat_obj[index_obj_all[idx_dist][0]]
                                    feat_neibor2 = feat_obj[index_obj_all[idx_dist][1]]
                                elif idx_dist.size == 1:
                                    feat_neibor1 = feat_obj[index_obj_all[idx_dist][0]]
                                    feat_neibor2 = np.zeros_like(feat_obj[0])
                                else:
                                    feat_neibor1 = np.zeros_like(feat_obj[0])
                                    feat_neibor2 = np.zeros_like(feat_obj[0])
                                feat_agent_.append(feat_neibor1)
                                feat_agent_.append(feat_neibor2)

                                feat_global = app_features[np.where(app_video_ids == int(video))[0][0]]
                                feat_global = feat_global.reshape(128,1024)
                                feat_global = feat_global[(t-1)*4]
                                feat_agent_.append(feat_global)  # shape:  5 1024

                                if len(feat_agent_)>0:
                                    feat_t.append(np.asarray(feat_agent_)) 
                                    index_t.append(index)  # index of feat bank you can use it to get frame id and obj id
                                else:
                                    print('no obj!')
                                    

                if len(feat_t) >0:
                    feat.append(feat_t) 
                    index_list.append(index_t)
                    nav_feat_idx.append(int(video))
                    counter +=1
        
        print('Total number of videos:', counter)


        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        obj_feat_id_to_index = {str(id): i for i, id in enumerate(nav_feat_idx)}
        self.app_feature_h5 = kwargs.pop('appearance_feat')

        self.dataset = VideoQADataset(answers,questions,ans_candidates,q_types,
                yes_indices, no_indices, video_ids, feat, index_list, nav_feat_idx, self.app_feature_h5, app_video_ids, q_ids,app_feat_id_to_index, obj_feat_id_to_index) 


        self.batch_size = kwargs['batch_size']

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    


class VideoQADataLoader(DataLoader):

    def __init__(self, **kwargs):    
        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))

        videolist_path = str(kwargs.pop('video_list'))
        with open(videolist_path, 'r') as f:
            data_list = f.read()
        nav_idx = np.fromstring(data_list[1:-1], dtype=int, sep=',')


        with open(question_pt_path, 'rb') as f: # extract the langugae features from clip_rn50
            obj = pickle.load(f)
            questions = obj['questions'][nav_idx]
            video_ids = nav_idx
            q_ids = nav_idx
            answers = obj['answers']
            ans_candidates = obj['ans_candidates']
      

        print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file: # get the glocal features 
            app_video_ids = app_features_file['ids'][()]
            app_features = app_features_file['appearance_features'][()]
            motion_features = app_features_file['motion_features'][()]

        nav_idx.sort()
        feat = []
        index_list = []
        nav_feat_idx = []

        object_feat_path = str(kwargs.pop('object_feat'))
        print('loading object feature from %s' % (object_feat_path))
        for i in range(len(nav_idx)):
            tmp_path = os.listdir(object_feat_path)
            tmp = []
            for j, id in enumerate(tmp_path):
                tmp.append(int(id))
            tmp = np.array(tmp)
            if np.where(tmp == nav_idx[i])[0].size > 0:
                with h5py.File(os.path.join(object_feat_path, str(nav_idx[i]), 'feat_obj.h5'), 'r') as feat_file:
                    bbox = feat_file['bbox'][()]
                    feat_obj = feat_file['feat_obj'][()]
                    frame_id = feat_file['frame_id'][()] #[0,0,1,1,2,2,10,10,31,31] frame_id_obj = frame_id[index]
                    object_id = feat_file['object_id'][()]
                    score = feat_file['score'][()]

                    test_frame = np.arange(0, 32)
                    mask = np.isin(test_frame, frame_id) 
                    if test_frame[mask].size == test_frame.size: 

                        feat_t = []
                        index_t = []

                        for t in range(1, frame_id.max()+1):
                            t_idx = np.nonzero(frame_id == t)[0] # index of the obj featues in this frame
                            t_idx_ = np.nonzero(frame_id == t-1)[0] # index of the obj featues in last frame
  
                            for index in t_idx: # get each obj
                                j =object_id[index]
                                feat_agent_ = []
                                # if np.where(object_id[t_idx_] == j) is not None:
                                if j in object_id[t_idx_]:
                                    feat_curr = feat_obj[index]
                                    feat_agent_.append(feat_curr)

                                    index_obj = t_idx_[np.where(object_id[t_idx_] == j)[0][0]]
                                    feat_his = feat_obj[index_obj]
                                    feat_agent_.append(feat_his)

                                    index_obj_all = t_idx_[np.where(t_idx_ != index_obj)[0]]
                                    dist = dist_bbox(bbox[index],bbox[index_obj_all])

                                    idx_dist = dist.argsort()
                                    if idx_dist.size >= 2:
                                        feat_neibor1 = feat_obj[index_obj_all[idx_dist][0]]
                                        feat_neibor2 = feat_obj[index_obj_all[idx_dist][1]]
                                    elif idx_dist.size == 1:
                                        feat_neibor1 = feat_obj[index_obj_all[idx_dist][0]]
                                        feat_neibor2 = np.zeros_like(feat_obj[0])
                                    else:
                                        feat_neibor1 = np.zeros_like(feat_obj[0])
                                        feat_neibor2 = np.zeros_like(feat_obj[0])
                                    feat_agent_.append(feat_neibor1)
                                    feat_agent_.append(feat_neibor2)

                                    feat_global = app_features[np.where(app_video_ids == nav_idx[i])[0][0]]
                                    feat_global = feat_global.reshape(128,1024)
                                    feat_global = feat_global[(t-1)*4]
                                    feat_agent_.append(feat_global)  # choose one frame  shape:  5 1024

                                    if len(feat_agent_)>0:
                                        feat_t.append(np.asarray(feat_agent_)) 
                                        index_t.append(index)  # index of feat bank you can use it to get frame id and obj id
                                    else:
                                        print('no obj')                         

                feat.append(feat_t) # 1264 x n_bb  
                index_list.append(index_t)
                nav_feat_idx.append(nav_idx[i])

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.batch_size = kwargs['batch_size']
        self.dataset = VideoQADatasetTrain(feat, index_list, nav_feat_idx,self.batch_size) 
        
        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)