from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from collections import OrderedDict
from multiprocessing import Process, Queue
import numpy as np
import sys
import logging
from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.ce.chexbert_eval import load_chexbert, compute_ce_metric
from pycocoevalcap.rouge.rouge import Rouge
from modules.metrics import compute_mlc
from pycocoevalcap.meteor.meteor import Meteor
import sys
from typing import List, Union, Iterable
from itertools import zip_longest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

Bleu_scorer = None
Rouge_scorer = None
Meteor_scorer = None



#
# RADGRAPH_PATH = '/extra/shilei/dataset/physionet.org/files/radgraph/1.0.0/models/model_checkpoint/model.tar.gz'
# cache_path = '/home/shilei/project/R2GenRL/RadGraph/result/IU'
# entities_path = os.path.join(cache_path, "entities_cache.json")
# relations_path = os.path.join(cache_path, "relations_cache.json")

def sentence_score(hypothesis: str, references: List[str], trace=0):
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)

    hypothesis = [hypothesis] * len(references)

    sentence_score = 0

    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1,
                              remove_subwords=False)

    sentence_score = np.mean(scores)

    if trace > 0:
        print(hypothesis, references, sentence_score)

    return sentence_score
def corpus_score(sys_stream: List[str],
                 ref_streams: Union[str, List[Iterable[str]]], trace=0):
    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]

    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]

    # print('l1 == ', len(sys_stream), 'l2 == ', len(ref_streams))

    fhs = [sys_stream] + ref_streams
    # print('fhs =====================', fhs)

    corpus_score = []
    for lines in zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")

        hypo, *refs = lines
        corpus_score.append(sentence_score(hypo, refs, trace=0))

    # corpus_score /= len(sys_stream)

    return np.array(corpus_score)


def init_scorer():
    global Bleu_scorer
    global Rouge_scorer
    global Meteor_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)
    Rouge_scorer = Rouge_scorer or Rouge()
    Meteor_scorer = Meteor_scorer or Meteor()


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def process_data(gts_report, gt_ids, gen_report, device, queue):
    # ���������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������
    gen_rad_entity_f1s, gen_rad_relation_f1s = cal_rad(gts_report, gt_ids, gen_report, device=device)
    queue.put((gen_rad_entity_f1s, gen_rad_relation_f1s))

def process_reports(chexbert_path, gen_report, greedy_report, gts_report):
    gen_label, greedy_label, gt_label = label(chexbert_path, gen_report, greedy_report, gts_report)
    return gen_label, greedy_label, gt_label

    # print(f"Processed data on device {device}")
def get_self_critical_reward(greedy_res, data_gts, gen_result, hyp, tokenizer, image_ids):
    gt_ids = []
    for id in image_ids:
        id = str(id).split('/')[-3:-1]
        gt_ids.append(id)
    batch_size = len(data_gts)
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts)  # gen_result_size  = batch_size * seq_per_img
    assert greedy_res.shape[0] == batch_size

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    gts = OrderedDict()
    data_gts = data_gts.cpu().numpy()

    gen_report = tokenizer.decode_batch(gen_result)
    greedy_report = tokenizer.decode_batch(greedy_res)
    gts_report = tokenizer.decode_batch(data_gts)
    # gts_move_report.append(gts_report)
    # print('gt:', len(gts_report))
    # print('gen:', len(gen_report))
    # print('greedy:', len(greedy_report))
    # gen_move_score = corpus_score(gen_report, gts_move_report)
    # greedy_move_score = corpus_score(greedy_report, gts_move_report)
    # print(gen_move_score)
    # print(greedy_move_score)
    # move_socre = np.append(gen_move_score, greedy_move_score)
    # chexbert_path = '/home/shilei/project/R2GenRL/CheXbert/checkpoint/chexbert.pth'
    # gen_label, greedy_label, gt_label = process_reports(chexbert_path, gen_report, greedy_report, gts_report)
    # gt_label = np.array(gt_label)
    # gen_label = np.array(gen_label)
    # greedy_label = np.array(greedy_label)
    # greedy_f1, greedy_recall, greedy_pre, _ = compute_mlc(gt_label, greedy_label)
    # gen_f1, gen_recall, gen_pre, _ = compute_mlc(gt_label, gen_label)
    # score_f1 = np.append(gen_f1, greedy_f1)
    # print('f1====', score_f1)
    # score_recall = np.append(gen_recall, greedy_recall)
    # score_pre = np.append(gen_pre, greedy_pre)
    # print('report=============================', gts_report)

    # half = len(gen_report) // 2
    # gt_ids_fr, gt_ids_ba = gt_ids[:half], gt_ids[half:]
    # gen_report_fr, gen_report_ba = gen_report[:half], gen_report[half:]
    # greedy_report_fr, greedy_report_ba = greedy_report[:half], greedy_report[half:]
    # gen_rad_entity_f1s_fr, gen_rad_relation_f1s_fr = cal_rad(gt_ids_fr, gen_report_fr, device=0)
    # # print('gt_ids_ba == ', gt_ids_ba, 'report_ba == ', gen_report_ba)
    # gen_rad_entity_f1s_ba, gen_rad_relation_f1s_ba = cal_rad(gt_ids_ba, gen_report_ba, device=1)
    #
    # greedy_rad_entity_f1s_fr, greedy_rad_relation_f1s_fr = cal_rad(gt_ids_fr, greedy_report_fr, device=2)
    # greedy_rad_entity_f1s_ba, greedy_rad_relation_f1s_ba = cal_rad(gt_ids_ba, greedy_report_ba, device=3)


    # gen_rad_entity_f1s = np.append(gen_rad_entity_f1s_fr, gen_rad_entity_f1s_ba)
    # greedy_rad_entity_f1s = np.append(greedy_rad_entity_f1s_fr, greedy_rad_entity_f1s_ba)

    # gen_rad_relation_f1s =np.append(gen_rad_relation_f1s_fr, gen_rad_relation_f1s_ba)
    # greedy_rad_relation_f1 = np.append(greedy_rad_relation_f1s_fr, greedy_rad_relation_f1s_ba)


    # ���������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������rad_graph������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������
    # gen_report.extend(greedy_report)
    # # print('ids================', gt_ids)

    # gen_rad_entity_f1s, gen_rad_relation_f1s = cal_rad(gts_report, gt_ids, gen_report,device=1)

    # rad_entity_f1 = gen_rad_entity_f1s

    # # print('f1====', rad_entity_f1)

    # rad_relation_f1 = gen_rad_relation_f1s
    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]


    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i])]
    res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    gts_.update({i + gen_result_size: gts[i] for i in range(batch_size)})
    # b1, b2, b3 = hyp[0], hyp[1], hyp[2]
    # b1, b2, b3, b4 = hyp[0], hyp[1], hyp[2], hyp[3]
    b1, b2 = hyp[0], hyp[1]
    _, bleu_scores = Bleu_scorer.compute_score(gts_, res__, verbose = 0)
    avg_rouge_score, np_rouge_score = Rouge_scorer.compute_score(gts_, res__)
    _, meteor_scores = Meteor_scorer.compute_score(gts_, res__)
    bleu_scores_1 = np.array(bleu_scores[0])
    bleu_scores_4 = np.array(bleu_scores[3])
    # ce_score =
    # logger.info('Bleu scores: {:.4f}.'.format(_[3]))
    # scores = bleu_scores_4
    # scores = b1 * bleu_scores_1 + b2 * bleu_scores_4 + b3 * np_rouge_score
    # scores = b1 * bleu_scores_4 + b2 * np_rouge_score
    # scores = move_socre
    scores1, scores2, scores3 = bleu_scores_1, bleu_scores_4, np_rouge_score
    # print('b3:', rad_relation_f1.size, 'b4:', rad_entity_f1.size, 'b1:', bleu_scores_1.size, 'rg:', np_rouge_score.size)
    # scores = b1 * bleu_scores_1 + b2 * np_rouge_score + b3 * rad_entity_f1 + b4 * rad_relation_f1
    # scores = b1 * score_f1 + b2 * score_recall + b3 * score_pre
    # scores = np_rouge_score
    # scores = b1 * bleu_scores_1 + b2 * np_rouge_score + b3 * rad_entity_f1
    # scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    scores1 = scores1[:gen_result_size].reshape(batch_size, seq_per_img) - scores1[-batch_size:][:, np.newaxis]
    scores2 = scores2[:gen_result_size].reshape(batch_size, seq_per_img) - scores2[-batch_size:][:, np.newaxis]
    scores3 = scores3[:gen_result_size].reshape(batch_size, seq_per_img) - scores3[-batch_size:][:, np.newaxis]
    # scores = scores.reshape(gen_result_size)

    # rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
    scores1 = scores1.reshape(gen_result_size)
    
    rewards1 = np.repeat(scores1[:, np.newaxis], gen_result.shape[1], 1)
    
    scores2 = scores2.reshape(gen_result_size)
    
    rewards2 = np.repeat(scores2[:, np.newaxis], gen_result.shape[1], 1)

    scores3 = scores3.reshape(gen_result_size)
    
    rewards3 = np.repeat(scores3[:, np.newaxis], gen_result.shape[1], 1)

    return rewards1, rewards2, rewards3
