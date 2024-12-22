from os.path import join as pjoin
import os
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import random
import codecs as cs
import json
import clip
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

class MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                if np.isnan(motion).any():
                    print(name)
                self.data.append(motion)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class Text2MotionDatasetEval(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]
        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        # print(line)
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list, length_list = new_name_list, length_list

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return caption, motion, m_length

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)


class FineText2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, clip_version='ViT-B/32', path=None, t2m_transformer=None):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.clip_model = None
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
 
        new_name_list = []
        length_list = []
        key_map = {'Left arm': 'left arm', 'Right arm': 'right arm', 'Left leg': 'left leg', 'Right leg': 'right leg', 'Spine': 'spine'}
        for name in tqdm(id_list):
            recaption_text_dict = {}
            # try:
            #     with cs.open(pjoin(opt.recaption_text_dir, name + '.json'), 'r', encoding='utf-8') as recaption_f:
            #         for recaption_line in recaption_f.readlines():
            #             data = json.loads(recaption_line.strip())
            #             recaption_text_dict[data["body part"]] = data["description"]

            #     if len(recaption_text_dict) != 5:
            #         recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            # except:
            #     recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            try:
                with cs.open(pjoin(opt.recaption_text_dir, name + '.json'), 'r', encoding='utf-8') as recaption_f:
                        data = json.load(recaption_f)
                        keys = list(data.keys())
                        for key in keys:
                            recaption_text_dict[key_map[key]] = data[key]
                if len(recaption_text_dict) != 5:
                    recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            except:
                recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            try: 
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue

                text_data = []
                # recaption_emb_data = []
                recaption_text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        # print(line)
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]

                        # try:
                        #     recaption_emb = np.load(pjoin(opt.recaption_text_dir, 'emb', name + '.npy'))
                        # except:
                        #     pass
                            # print(pjoin(opt.recaption_text_dir, 'emb', name + '.npy'),'not found')
                            # continue
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                            recaption_text_data.append(recaption_text_dict)
                            # recaption_emb_data.append(recaption_emb)
                        else:
                            try:
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict],
                                                    #    'recaption_emb':[recaption_emb],
                                                       'recaption_text':[recaption_text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data,
                                    #    'recaption_emb':recaption_emb_data,
                                       'recaption_text':recaption_text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list, length_list = new_name_list, length_list

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False)  # Must set jit=False for training
        # Added support for cpu
        if str(self.opt.device) != "cpu":
            clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
            # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu
        # Freeze CLIP weights
        clip_model.to(self.opt.device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model
    
    def encode_text(self, raw_text):
        device = next(self.clip_model.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text
    
    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        #motion, m_length, text_list, recaption_emb_list, recaption_list = data['motion'], data['length'], data['text'], data['recaption_emb'], data['recaption_text']
        motion, m_length, text_list, recaption_list = data['motion'], data['length'], data['text'], data['recaption_text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        recaption = random.choice(recaption_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return caption, motion, m_length, recaption

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)

class FineText2MotionDataset_beta(data.Dataset):
    def __init__(self, opt, mean, std, split_file, clip_version='ViT-B/32', path=None, t2m_transformer=None):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.clip_model = None
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        if not os.path.exists(path):
            self.initialize(opt,split_file, path, t2m_transformer, mean, std)
        else:
            name_list = []
            new_name_list = []
            length_list = []
            data_dict = {}
            with cs.open(pjoin(path,'name_list.txt'), 'r') as f:
                for line in f.readlines():
                    name_list.append(line.strip())
            for name in tqdm(name_list):
                data = np.load(pjoin(path, name + '.npz'), allow_pickle=True)
                data_dict[name] = {'motion': data['motion'],
                            'length': data['length'],
                            'text': data['text'],
                            'recaption_emb': data['recaption_emb'],
                            'recaption_text': data['recaption_text']}
                new_name_list.append(name)
                length_list.append(data['length'])
            self.mean = mean
            self.std = std
            self.length_arr = np.array(length_list)
            self.data_dict = data_dict
            self.name_list = new_name_list

    def initialize(self,opt,split_file,path,t2m_transformer, mean, std):
        t2m_transformer.to(opt.device)
        os.makedirs(path, exist_ok=True)
        data_dict = {}
        id_list = []
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        BODY_PARTS = ['left arm','right arm','left leg','right leg','spine']
        key_map = {'Left arm': 'left arm', 'Right arm': 'right arm', 'Left leg': 'left leg', 'Right leg': 'right leg', 'Spine': 'spine'}
        for name in tqdm(id_list):
            recaption_text_dict = {}
            # try:
            #     with cs.open(pjoin(opt.recaption_text_dir, name + '.json'), 'r', encoding='utf-8') as recaption_f:
            #         for recaption_line in recaption_f.readlines():
            #             data = json.loads(recaption_line.strip())
            #             recaption_text_dict[data["body part"]] = data["description"]

            #     if len(recaption_text_dict) != 5:
            #         recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            # except:
            #     recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            try:
                with cs.open(pjoin(opt.recaption_text_dir, name + '.json'), 'r', encoding='utf-8') as recaption_f:
                        data = json.load(recaption_f)
                        keys = list(data.keys())
                        for key in keys:
                            recaption_text_dict[key_map[key]] = data[key]
                if len(recaption_text_dict) != 5:
                    recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            except:
                print(pjoin(opt.recaption_text_dir, name + '.json'),'not found')
                recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            try: 
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue

                text_data = []
                recaption_emb_data = []
                recaption_text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        # print(line)
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        recaption = [line_split[0] for _ in range(4)]
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]

                        for idx in range(4):
                            key = BODY_PARTS[idx]
                            recaption[idx] = [recaption[idx] + ' ' + recaption_text_dict[key]]

                        cond_vector = []
                        for cond in recaption:
                            cond_vector_ = t2m_transformer.encode_text(cond)
                            cond_vector.append(cond_vector_)
                        recaption_emb = torch.cat(cond_vector, dim=0).detach().cpu().numpy()

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                            recaption_text_data.append(recaption_text_dict)
                            recaption_emb_data.append(recaption_emb)
                        else:
                            try:
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                    'length': len(n_motion),
                                                    'text':[text_dict],
                                                    'recaption_emb':[recaption_emb],
                                                    'recaption_text':[recaption_text_dict]}
                                np.savez(pjoin(path,new_name), **data_dict[new_name])
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'text': text_data,
                                    'recaption_emb':recaption_emb_data,
                                    'recaption_text':recaption_text_data}
                    np.savez(pjoin(path,name), **data_dict[name])
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                print(e)
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list, length_list = new_name_list, length_list

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        with open(pjoin(path,'name_list.txt'), 'w', encoding='utf-8') as file:
            for name in self.name_list:
                file.write(name + '\n')
    
    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False)  # Must set jit=False for training
        # Added support for cpu
        if str(self.opt.device) != "cpu":
            clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
            # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu
        # Freeze CLIP weights
        clip_model.to(self.opt.device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model
    
    def encode_text(self, raw_text):
        device = next(self.clip_model.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text
    
    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, recaption_emb_list, recaption_list = data['motion'], data['length'], data['text'], data['recaption_emb'], data['recaption_text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        recaption_emb = random.choice(recaption_emb_list)
        recaption = random.choice(recaption_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return caption, motion, m_length, recaption_emb, recaption

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)

class FineText2MotionDatasetEval(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]

        new_name_list = []
        length_list = []
        key_map = {'Left arm': 'left arm', 'Right arm': 'right arm', 'Left leg': 'left leg', 'Right leg': 'right leg', 'Spine': 'spine'}
        # from itertools import islice
        # for name in tqdm(islice(id_list, 0, 100)):
        for name in tqdm(id_list):
            recaption_text_dict = {}
            # try:
            #     with cs.open(pjoin(opt.recaption_text_dir, name + '.json'), 'r', encoding='utf-8') as recaption_f:
            #         for recaption_line in recaption_f.readlines():
            #             data = json.loads(recaption_line.strip())
            #             recaption_text_dict[data["body part"]] = data["description"]

            #     if len(recaption_text_dict) != 5:
            #         print(name, 'recaption_text not equal 5')
            #         recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            # except:
            #     print(name, 'recaption_text not found')
            #     recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            try:
                with cs.open(pjoin(opt.recaption_text_dir, name + '.json'), 'r', encoding='utf-8') as recaption_f:
                        data = json.load(recaption_f)
                        keys = list(data.keys())
                        for key in keys:
                            recaption_text_dict[key_map[key]] = data[key]
                if len(recaption_text_dict) != 5:
                    recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            except:
                print(name, 'recaption_text not found')
                recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue

                text_data = []
                # recaption_emb_data = []
                recaption_text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        #TODO 
                        # try:
                        #     recaption_emb = np.load(pjoin(opt.recaption_text_dir, 'emb', name + '.npy'))
                        # except:
                        #     print(name,'recaption_emb not found')
                            # continue
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                            recaption_text_data.append(recaption_text_dict)
                            # recaption_emb_data.append(recaption_emb)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict],
                                                    #    'recaption_emb':[recaption_emb],
                                                       'recaption_text':[recaption_text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data,
                                    #    'recaption_emb':recaption_emb_data,
                                       'recaption_text':recaption_text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        #motion, m_length, text_list, recaption_emb_list, recaption_list = data['motion'], data['length'], data['text'], data['recaption_emb'], data['recaption_text']
        motion, m_length, text_list, recaption_list = data['motion'], data['length'], data['text'], data['recaption_text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        recaption = random.choice(recaption_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), recaption

class FineText2MotionDatasetEval_beta(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, path=None, t2m_transformer=None):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        if not os.path.exists(path):
            self.initialize(opt,split_file, path, t2m_transformer, mean, std)
        else:
            name_list = []
            new_name_list = []
            length_list = []
            data_dict = {}
            with cs.open(pjoin(path,'name_list.txt'), 'r') as f:
                for line in f.readlines():
                    name_list.append(line.strip())
            for name in tqdm(name_list):
                data = np.load(pjoin(path, name + '.npz'), allow_pickle=True)
                data_dict[name] = {'motion': data['motion'],
                            'length': data['length'],
                            'text': data['text'],
                            'recaption_emb': data['recaption_emb'],
                            'recaption_text': data['recaption_text']}
                new_name_list.append(name)
                length_list.append(data['length'])
            self.mean = mean
            self.std = std
            self.length_arr = np.array(length_list)
            self.data_dict = data_dict
            self.name_list = new_name_list
            self.reset_max_len(self.max_length)

    def initialize(self,opt,split_file,path,t2m_transformer, mean, std):
        print('Initializing dataset {}'.format(path))
        t2m_transformer.to(opt.device)
        os.makedirs(path, exist_ok=True)
        data_dict = {}
        id_list = []
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        BODY_PARTS = ['left arm','right arm','left leg','right leg','spine']
        key_map = {'Left arm': 'left arm', 'Right arm': 'right arm', 'Left leg': 'left leg', 'Right leg': 'right leg', 'Spine': 'spine'}
        for name in tqdm(id_list):
            recaption_text_dict = {}
            # try:
            #     with cs.open(pjoin(opt.recaption_text_dir, name + '.json'), 'r', encoding='utf-8') as recaption_f:
            #         for recaption_line in recaption_f.readlines():
            #             data = json.loads(recaption_line.strip())
            #             recaption_text_dict[data["body part"]] = data["description"]

            #     if len(recaption_text_dict) != 5:
            #         recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            # except:
            #     recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            try:
                with cs.open(pjoin(opt.recaption_text_dir, name + '.json'), 'r', encoding='utf-8') as recaption_f:
                        data = json.load(recaption_f)
                        keys = list(data.keys())
                        for key in keys:
                            recaption_text_dict[key_map[key]] = data[key]
                if len(recaption_text_dict) != 5:
                    recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            except:
                print(pjoin(opt.recaption_text_dir, name + '.json'),'not found')
                recaption_text_dict = {'left arm': '', 'right arm': '', 'left leg': '', 'right leg': '', 'spine': ''}
            # try: 
            motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
            if (len(motion)) < min_motion_len or (len(motion) >= 200):
                continue

            text_data = []
            recaption_emb_data = []
            recaption_text_data = []
            flag = False
            with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split('#')
                    # print(line)
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    recaption = [line_split[0] for _ in range(4)]
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = tokens
                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]

                    for idx in range(4):
                        key = BODY_PARTS[idx]
                        recaption[idx] = [recaption[idx] + ' ' + recaption_text_dict[key]]

                    cond_vector = []
                    for cond in recaption:
                        cond_vector_ = t2m_transformer.encode_text(cond)
                        cond_vector.append(cond_vector_)
                    recaption_emb = torch.cat(cond_vector, dim=0).detach().cpu().numpy()

                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                        recaption_text_data.append(recaption_text_dict)
                        recaption_emb_data.append(recaption_emb)
                    else:
                        try:
                            if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                continue
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            while new_name in data_dict:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            data_dict[new_name] = {'motion': n_motion,
                                                'length': len(n_motion),
                                                'text':[text_dict],
                                                'recaption_emb':[recaption_emb],
                                                'recaption_text':[recaption_text_dict]}
                            np.savez(pjoin(path,new_name), **data_dict[new_name])
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))
                        except:
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag, to_tag, name)
                            # break

            if flag:
                data_dict[name] = {'motion': motion,
                                'length': len(motion),
                                'text': text_data,
                                'recaption_emb':recaption_emb_data,
                                'recaption_text':recaption_text_data}
                np.savez(pjoin(path,name), **data_dict[name])
                new_name_list.append(name)
                length_list.append(len(motion))
            # except Exception as e:
            #     print(e)
            #     pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list, length_list = new_name_list, length_list

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        with open(pjoin(path,'name_list.txt'), 'w', encoding='utf-8') as file:
            for name in self.name_list:
                file.write(name + '\n')

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, recaption_emb_list, recaption_list = data['motion'], data['length'], data['text'], data['recaption_emb'], data['recaption_text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        recaption_emb = random.choice(recaption_emb_list)
        recaption = random.choice(recaption_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), recaption_emb, recaption
