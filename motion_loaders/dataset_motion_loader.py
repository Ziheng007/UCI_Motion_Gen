from data.t2m_dataset import Text2MotionDatasetEval,FineText2MotionDatasetEval, FineText2MotionDatasetEval_beta,collate_fn # TODO
from utils.word_vectorizer import WordVectorizer
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from utils.get_opt import get_opt

def get_dataset_motion_loader_beta(opt_path, batch_size, fname, device, is_recap=False, recap_name=None, t2m_transformer=None):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        print('Loading dataset %s ...' % opt.dataset_name)
        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        if not is_recap:
            dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer)
        else:
            opt.recaption_text_dir = pjoin(opt.data_root, recap_name)
            print('Recaption_text_dir:', opt.recaption_text_dir)
            # dataset = FineText2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer)
            dataset = FineText2MotionDatasetEval_beta(opt, mean, std, split_file, w_vectorizer, path=pjoin(opt.data_root, recap_name,f'eval_{fname}'), t2m_transformer=t2m_transformer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset

def get_dataset_motion_loader(opt_path, batch_size, fname, device, is_recap=False, recap_name=None, t2m_transformer=None):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        print('Loading dataset %s ...' % opt.dataset_name)
        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        if not is_recap:
            dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer)
        else:
            opt.recaption_text_dir = pjoin(opt.data_root, recap_name)
            print('Recaption_text_dir:', opt.recaption_text_dir)
            dataset = FineText2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset

