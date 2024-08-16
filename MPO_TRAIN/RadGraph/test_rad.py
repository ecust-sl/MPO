
from radgraph_evaluate_model import run_radgraph

import os
import json
RADGRAPH_PATH = '/extra/shilei/dataset/physionet.org/files/radgraph/1.0.0/models/model_checkpoint/model.tar.gz'
cache_path = '/home/shilei/project/R2GenRL/RadGraph/result/IU'
entities_path = os.path.join(cache_path, "entities_cache.json")
relations_path = os.path.join(cache_path, "relations_cache.json")
gt_report = ['AP semi upright view of the chest provided.There is no focal consolidation, effusion, or pneumothorax.  Bibasilar atelectasis is similar to prior.  Mild cardiomegaly and large hiatal hernia are similar to prior. Imaged osseous structures are intact.  No free air below the right hemidiaphragm is seen.No acute intrathoracic process.']
cache_gt_csv = [['p20000065', 's51613820']]
cache_pred_csv = ['is a semi','AP semi upright view of the chest provided.There is no focal consolidation, effusion, or pneumothorax.  Bibasilar atelectasis is']
def cal_rad(gt_report, gt, pred, device = 1):


    entity_f1, relation_f1 = run_radgraph(gt_report, gt, pred, cache_path, RADGRAPH_PATH,
                 entities_path, relations_path, device)

    # print('entity_f1 ==',entity_f1)
    #
    # print('relation_f1 ==', relation_f1)
    return entity_f1, relation_f1


if __name__ == "__main__":
    cal_rad(gt_report, cache_gt_csv, cache_pred_csv)


