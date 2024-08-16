from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


def compute_mlc(gt, pred):
    res_mlc = {}
    res_mlc2 = {}
    # avg_aucroc = 0
    # for i, label in enumerate(label_set):
    #     res_mlc['AUCROC_' + label] = roc_auc_score(gt[:, i], pred[:, i])
    #     avg_aucroc += res_mlc['AUCROC_' + label]
    # res_mlc['AVG_AUCROC'] = avg_aucroc / len(label_set)
    batch_size = gt.shape[0]
    res_mlc['F1_MACRO'] = []
    res_mlc['F1_MICRO'] = []
    res_mlc['RECALL_MACRO'] = []
    res_mlc['RECALL_MICRO'] = []
    res_mlc['PRECISION_MACRO'] = []
    res_mlc['PRECISION_MICRO'] = []
    for i in range(batch_size):
        gt_ = gt[i]
        pred_ = pred[i]
        res_mlc['F1_MACRO'].append(f1_score(gt_, pred_, average="macro"))
        res_mlc['F1_MICRO'].append(f1_score(gt_, pred_, average="micro"))
        res_mlc['RECALL_MACRO'].append(recall_score(gt_, pred_, average="macro"))
        res_mlc['RECALL_MICRO'].append(recall_score(gt_, pred_, average="micro"))
        res_mlc['PRECISION_MACRO'].append(precision_score(gt_, pred_, average="macro"))
        res_mlc['PRECISION_MICRO'] += precision_score(gt_, pred_, average="micro")
    #
    # res_mlc['F1_MACRO'] /= batch_size
    # res_mlc['F1_MICRO'] /= batch_size
    # res_mlc['RECALL_MACRO'] /= batch_size
    # res_mlc['RECALL_MICRO'] /= batch_size
    # res_mlc['PRECISION_MACRO'] /= batch_size
    # res_mlc['PRECISION_MICRO'] /= batch_size
    res_mlc2['F1_MACRO'] = f1_score(gt, pred, average="macro")
    res_mlc2['F1_MICRO'] = f1_score(gt, pred, average="micro")
    res_mlc2['RECALL_MACRO'] = recall_score(gt, pred, average="macro")
    res_mlc2['RECALL_MICRO'] = recall_score(gt, pred, average="micro")
    res_mlc2['PRECISION_MACRO'] = precision_score(gt, pred, average="macro")
    res_mlc2['PRECISION_MICRO'] = precision_score(gt, pred, average="micro")

    return res_mlc['F1_MACRO'], res_mlc['RECALL_MACRO'],res_mlc['PRECISION_MACRO'], res_mlc2


class MetricWrapper(object):
    def __init__(self, label_set):
        self.label_set = label_set

    def __call__(self, gts, res, gts_mlc, res_mlc):
        eval_res = compute_scores(gts, res)
        eval_res_mlc = compute_mlc(gts_mlc, res_mlc, self.label_set)

        eval_res.update(**eval_res_mlc)
        return eval_res
