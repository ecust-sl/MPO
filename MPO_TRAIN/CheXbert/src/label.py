import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
# import utils import *
from CheXbert.src import utils
from CheXbert.src.models.bert_labeler import bert_labeler
from CheXbert.src.bert_tokenizer import tokenize
from transformers import BertTokenizer
from collections import OrderedDict
from CheXbert.src.datasets.unlabeled_dataset import UnlabeledDataset
from CheXbert.src.constants import *
from tqdm import tqdm

def collate_fn_no_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len,
       where the reports have no associated labels
    @param sample_list (List): A list of samples. Each sample is a dictionary with
                               keys 'imp', 'len' as returned by the __getitem__
                               function of ImpressionsDataset

    @returns batch (dictionary): A dictionary with keys 'imp' and 'len' but now
                                 'imp' is a tensor with padding and batch size as the
                                 first dimension. 'len' is a list of the length of 
                                 each sequence in batch
    """
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                  batch_first=True,
                                                  padding_value=PAD_IDX)
    len_list = [s['len'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list}
    return batch

def load_unlabeled_data(gen_report, greedy_report, gt_report, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=False):
    """ Create UnlabeledDataset object for the input reports
    @param csv_path (string): path to csv file containing reports
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not  
    
    @returns loader (dataloader): dataloader object for the reports
    """
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(gen_report)
    dset_1 = UnlabeledDataset(greedy_report)
    dset_2 = UnlabeledDataset(gt_report)


    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    loader_1 = torch.utils.data.DataLoader(dset_1, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    loader_2 = torch.utils.data.DataLoader(dset_2, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    return loader, loader_1, loader_2


def load_unlabeled_data_test(gen_report, gt_report, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=False):
    """ Create UnlabeledDataset object for the input reports
    @param csv_path (string): path to csv file containing reports
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not

    @returns loader (dataloader): dataloader object for the reports
    """
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(gen_report)
    dset_1 = UnlabeledDataset(gt_report)
    # dset_2 = UnlabeledDataset(gt_report)

    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    loader_1 = torch.utils.data.DataLoader(dset_1, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, collate_fn=collate_fn)
    # loader_2 = torch.utils.data.DataLoader(dset_2, batch_size=batch_size, shuffle=shuffle,
    #                                        num_workers=num_workers, collate_fn=collate_fn)
    return loader, loader_1
def label(checkpoint_path, gen_report, greedy_report, gt_report):
    path_chex = '/extra/shilei/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv'
    """Labels a dataset of reports
    @param checkpoint_path (string): location of saved model checkpoint 
    @param csv_path (string): location of csv with reports

    @returns y_pred (List[List[int]]): Labels for each of the 14 conditions, per report  
    """
    ld, ld_1, ld_2 = load_unlabeled_data(gen_report, greedy_report, gt_report)
    # gt_ids = [[id[1:] for id in sublist] for sublist in gt_ids]
    # print('ids == ', gt_ids)
    model = bert_labeler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0: #works even if only 1 GPU available
        # print("Using", torch.cuda.device_count(), "GPUs!")
        # model = nn.DataParallel(model) #to utilize multiple GPU's
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]
    y_pred_1 = [[] for _ in range(len(CONDITIONS))]
    y_pred_2 = [[] for _ in range(len(CONDITIONS))]


    # print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    # print("The batch size is %d" % BATCH_SIZE)
    with torch.no_grad():
        for i, data in enumerate(tqdm(ld, disable=True)):
            batch = data['imp'] #(batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)

            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1) #shape is (batch_size)
                y_pred[j].append(curr_y_pred)

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)
        # print('pred=== ', y_pred)

        for i, data in enumerate(tqdm(ld_1, disable=True)):
            batch = data['imp'] #(batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)

            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1) #shape is (batch_size)
                y_pred_1[j].append(curr_y_pred)

        for j in range(len(y_pred_1)):
            y_pred_1[j] = torch.cat(y_pred_1[j], dim=0)

        for i, data in enumerate(tqdm(ld_2, disable=True)):
            batch = data['imp']  # (batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)

            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1)  # shape is (batch_size)
                y_pred_2[j].append(curr_y_pred)

        for j in range(len(y_pred_2)):
            y_pred_2[j] = torch.cat(y_pred_2[j], dim=0)

    if was_training:
        model.train()

    y_pred = [t.tolist() for t in y_pred]
    y_pred = np.array(y_pred)
    y_pred = y_pred.T
    # 示例数组
    # y_pred = np.array([np.nan, 2, 3, np.nan, 1, 3, 2, 4])

    # 第一步：将 np.nan 替换为 0
    y_pred = np.nan_to_num(y_pred, nan=0)

    # 第二步：使用 np.where 替换 3 为 1，2 为 0
    y_pred = np.where(y_pred == 3, 0, y_pred)
    y_pred = np.where(y_pred == 2, 0, y_pred)


    y_pred_1 = [t.tolist() for t in y_pred_1]
    y_pred_1 = np.array(y_pred_1)
    y_pred_1 = y_pred_1.T
    # 示例数组
    # y_pred = np.array([np.nan, 2, 3, np.nan, 1, 3, 2, 4])

    # 第一步：将 np.nan 替换为 0
    y_pred_1 = np.nan_to_num(y_pred_1, nan=0)

    # 第二步：使用 np.where 替换 3 为 1，2 为 0
    y_pred_1 = np.where(y_pred_1 == 3, 0, y_pred_1)
    y_pred_1 = np.where(y_pred_1 == 2, 0, y_pred_1)

    y_pred_2 = [t.tolist() for t in y_pred_2]
    y_pred_2 = np.array(y_pred_2)
    y_pred_2 = y_pred_2.T
    # 示例数组
    # y_pred = np.array([np.nan, 2, 3, np.nan, 1, 3, 2, 4])

    # 第一步：将 np.nan 替换为 0
    y_pred_2 = np.nan_to_num(y_pred_2, nan=0)

    # 第二步：使用 np.where 替换 3 为 1，2 为 0
    y_pred_2 = np.where(y_pred_2 == 3, 0, y_pred_2)
    y_pred_2 = np.where(y_pred_2 == 2, 0, y_pred_2)
    return y_pred, y_pred_1, y_pred_2


def label_test(checkpoint_path, gen_report, gt_reports):
    path_chex = '/extra/shilei/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv'
    """Labels a dataset of reports
    @param checkpoint_path (string): location of saved model checkpoint 
    @param csv_path (string): location of csv with reports

    @returns y_pred (List[List[int]]): Labels for each of the 14 conditions, per report  
    """
    ld,ld_1 = load_unlabeled_data_test(gen_report, gt_reports)
    # gt_ids = [[id[1:] for id in sublist] for sublist in gt_ids]
    # print('ids == ', gt_ids)
    model = bert_labeler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:  # works even if only 1 GPU available
        # print("Using", torch.cuda.device_count(), "GPUs!")
        # model = nn.DataParallel(model) #to utilize multiple GPU's
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location= torch.device("cuda"))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]
    y_pred_2 = [[] for _ in range(len(CONDITIONS))]
    # y_pred_2 = [[] for _ in range(len(CONDITIONS))]

    # print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    # print("The batch size is %d" % BATCH_SIZE)
    with torch.no_grad():
        for i, data in enumerate(tqdm(ld, disable=True)):
            batch = data['imp']  # (batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)

            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1)  # shape is (batch_size)
                y_pred[j].append(curr_y_pred)

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)


        for i, data in enumerate(tqdm(ld_1, disable=True)):
            batch = data['imp']  # (batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)

            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1)  # shape is (batch_size)
                y_pred_2[j].append(curr_y_pred)

        for j in range(len(y_pred_2)):
            y_pred_2[j] = torch.cat(y_pred_2[j], dim=0)
        # print('pred=== ', y_pred)

    # 返回结
    # print(y_pred_2)

    if was_training:
        model.train()

    y_pred = [t.tolist() for t in y_pred]
    y_pred = np.array(y_pred)
    y_pred = y_pred.T
    # 示例数组
    # y_pred = np.array([np.nan, 2, 3, np.nan, 1, 3, 2, 4])

    # 第一步：将 np.nan 替换为 0
    y_pred = np.nan_to_num(y_pred, nan=0)
    #
    # # 第二步：使用 np.where 替换 3 为 1，2 为 0
    y_pred = np.where(y_pred == 3, 0, y_pred)
    y_pred = np.where(y_pred == 2, 0, y_pred)


    y_pred_2 = [t.tolist() for t in y_pred_2]
    y_pred_2 = np.array(y_pred_2)
    y_pred_2 = y_pred_2.T
    # 示例数组
    # y_pred = np.array([np.nan, 2, 3, np.nan, 1, 3, 2, 4])

    # 第一步：将 np.nan 替换为 0
    y_pred_2 = np.nan_to_num(y_pred_2, nan=0)
    #
    # # 第二步：使用 np.where 替换 3 为 1，2 为 0
    y_pred_2 = np.where(y_pred_2 == 3, 0, y_pred_2)
    y_pred_2 = np.where(y_pred_2 == 2, 0, y_pred_2)
    # out_path = "/home/shilei/project/R2GenRL/CheXbert/src/result"
    # save_preds(y_pred, os.path.join(out_path, "labeled_gens.csv"))
    # save_preds(y_pred_2, os.path.join(out_path, "labeled_gts.csv"))

    return y_pred, y_pred_2


def save_preds(y_pred, out_path):
    """Save predictions as out_path/labeled_reports.csv
    @param y_pred (List[List[int]]): list of predictions for each report
    @param csv_path (string): path to csv containing reports
    @param out_path (string): path to output directory
    """
    y_pred = np.array(y_pred)
    y_pred = y_pred.T

    df = pd.DataFrame(y_pred, columns=CONDITIONS)
    new_cols = CONDITIONS
    df = df[new_cols]

    df.replace(0, np.nan, inplace=True)  # blank class is NaN
    df.replace(3, -1, inplace=True)  # uncertain class is -1
    df.replace(2, 0, inplace=True)  # negative class is 0

    df.to_csv(out_path, index=False)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    parser = argparse.ArgumentParser(description='Label a csv file containing radiology reports')
    parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                        help='path to csv containing reports. The reports should be \
                              under the \"Report Impression\" column', default='sample_reports.csv')
    parser.add_argument('-o', '--output_dir', type=str, nargs='?', required=True,
                        help='path to intended output folder', default='labeled_erports.csv')
    parser.add_argument('-c', '--checkpoint', type=str, nargs='?', required=True,
                        help='path to the pytorch checkpoint', default='/home/shilei/project/CheXbert/checkpoint/chexbert.pth')
    args = parser.parse_args()
    csv_path = args.data
    out_path = args.output_dir
    checkpoint_path = args.checkpoint

    y_pred = label(checkpoint_path, reports)
    save_preds(y_pred)
