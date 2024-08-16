import math
import os
import glob
import json
import pandas as pd
import re
from tqdm import tqdm
import argparse

"""Code adapted from https://physionet.org/content/radgraph/1.0.0: models/inference.py."""
#病变实体 Cardiomediastinum等 方位实体 right left等
def get_key_entity():
    key_relation_entities = [
        'Atelectasis',
        'atelectatic',#肺不张
        'Cardiomegaly',
        'Consolidation',
        'Consolidations',
        'Edema',
        'Cardiomediastinum',
        'Fracture',
        'Lungs',
        'lung',
        'Lesion',
        'Opacity',
        'opacities',
        'Pleural',
        'Effusion',
        'Effusions',
        'Pneumonia',
        'Pneumothorax',
        'pneumothoraces',#气胸
        'devices',
        'support'
    ]
    lowercase_entities = [entity.lower() for entity in key_relation_entities]

    return lowercase_entities

def preprocess_reports(data_path, start, end, sentence=False, image=False, ids = None):
    """ Load up the files mentioned in the temporary json file, and
    processes them in format that the dygie model can take as input.
    Also save the processed file in a temporary file.
    """
    # impressions = pd.read_csv(data_path)
    # if start != None and end != None:
    #     impressions = impressions.iloc[start:end]
    # final_list = []
    # for idx, row in impressions.iterrows():
    #     print('type == ', type(row['report']))
    #     if (isinstance(row["report"], float) and math.isnan(row["report"])): continue
    #     sen = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ', row["report"]).split()
    #     # print('sen ==' ,sen)
    #     temp_dict = {}
    #
    #     if not sentence:  # Report-level
    #         if image:  # Different image can have different reports
    #             temp_dict["doc_key"] = f"{row['dicom_id']}_{row['study_id']}"
    #         else:
    #             temp_dict["doc_key"] = str(row["study_id"])
    #     else:  # Sentence-level
    #         temp_dict["doc_key"] = f"{row['study_id']}_{row['sentence_id']}"
    #
    #     ## Current way of inference takes in the whole report as 1 sentence
    #     temp_dict["sentences"] = [sen]
    #
    #     final_list.append(temp_dict)
    #
    #     if(idx % 1000 == 0):
    #         print(f"{idx+1} reports done")
    #     在这里的话data_path 被我等价为 data
    impressions = data_path
    if start != None and end != None:
        impressions = data.iloc[start:end]
    final_list = []
    # print('imepressions ===============', impressions)

    for idx, row in enumerate(impressions):
        # print('row == ', row)
        temp_dict = {}
        # id = ids[idx]
        id_pair = ids[idx]
        id = f"{id_pair[0][:3]}/{id_pair[0]}/{id_pair[1]}.txt"
        # print('id == ', id)
        if (isinstance(row, float) and math.isnan(row)): continue
        if not sentence:  # Report-level
            if image:  # Different image can have different reports
                        temp_dict["doc_key"] = f"{row['dicom_id']}_{row['study_id']}"
            else:
                temp_dict["doc_key"] = str(id)
        else:  # Sentence-level
            temp_dict["doc_key"] = str(id)
        sen = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ', row).split()
        # print('sen ==' ,sen)


        ## Current way of inference takes in the whole report as 1 sentence
        temp_dict["sentences"] = [sen]

        final_list.append(temp_dict)

        # if (idx % 1000 == 0):
        #     print(f"{idx + 1} reports done")
    
    # print(f"{idx+1} reports done")
    
    with open("/home/shilei/project/R2GenRL/RadGraph/temp_data/temp_dygie_input.json",'w') as outfile:
        for item in final_list:
            json.dump(item, outfile)
            outfile.write("\n")

def run_inference(model_path, cuda):
    
    """ Runs the inference on the processed input files. Saves the result in a
    temporary output file
    
    Args:
        model_path: Path to the model checkpoint
        cuda: GPU id
    
    
    """
    out_path = "/home/shilei/project/R2GenRL/RadGraph/temp_data/temp_dygie_output.json"
    data_path = "/home/shilei/project/R2GenRL/RadGraph/temp_data/temp_dygie_input.json"
    
    os.system(f"allennlp predict {model_path} \
             {data_path} \
            --predictor dygie \
            --include-package dygie \
            --use-dataset-reader \
            --output-file {out_path} \
            --cuda-device {cuda} \
            --batch-size 1 \
            --silent")

def postprocess_reports(data_source, data_split):
    
    """Post processes all the reports and saves the result in train.json format
    """
    final_dict = {}

    file_name = f"/home/shilei/project/R2GenRL/RadGraph/temp_data/temp_dygie_output.json"
    data = []


    with open(file_name,'r') as f:
        for line in f:
            data.append(json.loads(line))

    for file in data:
        postprocess_individual_report(file, final_dict, data_source=data_source, data_split=data_split)
    
    return final_dict

def postprocess_individual_report(file, final_dict, data_source=None, data_split="inference"):
    
    """Postprocesses individual report
    
    Args:
        file: output dict for individual reports
        final_dict: Dict for storing all the reports
    """
    
    try:
        temp_dict = {}

        temp_dict['text'] = " ".join(file['sentences'][0])
        n = file['predicted_ner'][0]
        r = file['predicted_relations'][0]
        s = file['sentences'][0]
        temp_dict["entities"] = get_entity(n,r,s)
        temp_dict["data_source"] = data_source
        temp_dict["data_split"] = data_split

        if file['doc_key'] in final_dict:  # Handle duplicate study IDs.
            final_dict[file['doc_key'] + '+'] = temp_dict
        else:
            final_dict[file['doc_key']] = temp_dict
    
    except:
        print(f"Error in doc key: {file['doc_key']}. Skipping inference on this file")
        
# def get_entity(n,r,s):
#
#     """Gets the entities for individual reports
#
#     Args:
#         n: list of entities in the report
#         r: list of relations in the report
#         s: list containing tokens of the sentence
#
#     Returns:
#         dict_entity: Dictionary containing the entites in the format similar to train.json
#
#     """
#
#     dict_entity = {}
#     rel_list = [item[0:2] for item in r]
#     print('rel_list == ',rel_list)
#     ner_list = [item[0:2] for item in n]
#     all_tokens = {}
#     key_token_pos = {}
#     for idx, item in enumerate(n):
#         start_idx, end_idx, label = item[0], item[1], item[2]
#         tok = " ".join(s[start_idx:end_idx+1]).lower()
#         all_tokens[str(idx + 1)] = tok
#     key_entities = get_key_entity()
#     for idx, item in enumerate(n):
#         temp_dict = {}
#         vis_en = 0
#         start_idx, end_idx, label = item[0], item[1], item[2]
#         tok = " ".join(s[start_idx:end_idx+1]).lower()
#         if tok in key_entities:
#             vis_en = 1
#
#         temp_dict['tokens'] = " ".join(s[start_idx:end_idx+1])
#         temp_dict['label'] = label
#         temp_dict['start_ix'] = start_idx
#         temp_dict['end_ix'] = end_idx
#         rel = []
#
#         relation_idx = [i for i,val in enumerate(rel_list) if val== [start_idx, end_idx]]
#         for i,val in enumerate(relation_idx):
#             obj = r[val][2:4]
#             lab = r[val][4]
#             try:
#                 object_idx = ner_list.index(obj) + 1
#             except:
#                 continue
#             tmp_en = all_tokens[str(object_idx)]
#             if tmp_en in key_entities or vis_en:
#                 rel.append([lab, str(object_idx)])
#
#
#         temp_dict['relations'] = rel
#         dict_entity[str(idx + 1)] = temp_dict
#         to_delete = []
#         for key, value in dict_entity.items():
#             if value['tokens'].lower() not in key_entities and not value['relations']:
#                 # 检查是否有其他条目的relations包含这个key
#                 is_related = any(key in [rel[1] for rel in v['relations']] for v in dict_entity.values())
#                 if not is_related:
#                     to_delete.append(key)
#
#         # 步骤2: 删除这些键
#         for key in to_delete:
#             del dict_entity[key]
#     print('dict ==== ', dict_entity)
#     return dict_entity


def get_entity(n, r, s):
    """Gets the entities for individual reports

    Args:
        n: list of entities in the report
        r: list of relations in the report
        s: list containing tokens of the sentence

    Returns:
        dict_entity: Dictionary containing the entites in the format similar to train.json

    """

    dict_entity = {}
    rel_list = [item[0:2] for item in r]
    ner_list = [item[0:2] for item in n]
    key_tokens = get_key_entity()
    for idx, item in enumerate(n):
        temp_dict = {}
        start_idx, end_idx, label = item[0], item[1], item[2]
        temp_dict['tokens'] = " ".join(s[start_idx:end_idx + 1])
        temp_dict['label'] = label
        temp_dict['start_ix'] = start_idx
        temp_dict['end_ix'] = end_idx
        rel = []
        relation_idx = [i for i, val in enumerate(rel_list) if val == [start_idx, end_idx]]
        for i, val in enumerate(relation_idx):
            obj = r[val][2:4]
            lab = r[val][4]
            try:
                object_idx = ner_list.index(obj) + 1
            except:
                continue
            rel.append([lab, str(object_idx)])
        temp_dict['relations'] = rel
        dict_entity[str(idx + 1)] = temp_dict

    for key, value in dict_entity.items():
        new_relations = []
        for rel in value['relations']:
            # 检查relations中的第二列对应的token和当前token是否都不在key_tokens里面
            if dict_entity[rel[1]]['tokens'].lower() in key_tokens or value['tokens'].lower() in key_tokens:
                new_relations.append(rel)
        dict_entity[key]['relations'] = new_relations

    # 步骤2: 找到需要删除的键
    to_delete = []
    for key, value in dict_entity.items():
        if value['tokens'].lower() not in key_tokens and not value['relations']:
            # 检查是否有其他条目的relations包含这个key
            is_related = any(key in [rel[1] for rel in v['relations']] for v in dict_entity.values())
            if not is_related:
                to_delete.append(key)

    # 步骤3: 删除这些键
    for key in to_delete:
        del dict_entity[key]


    return dict_entity
def cleanup():
    """Removes all the temporary files created during the inference process
    
    """
    # os.system("rm temp_file_list.json")
    # os.system("rm temp_data/temp_dygie_input.json")
    # os.system("rm temp_data/temp_dygie_output.json")

def _json_to_csv(path, csv_path):
    with open(path, "r") as f:
        match_results = json.load(f)
    reconstructed_reports = []
    for _, (_, train, match) in match_results.items():
        test_report_id = match[0][0][:8]
        reconstructed_reports.append((test_report_id, train))
    pd.DataFrame(reconstructed_reports, columns=["study_id", "report"]).to_csv(csv_path)

def _add_ids_column(
            csv_path, study_id_csv_path, output_path):
    with open(csv_path, "r") as f:
        generated_reports = pd.read_csv(f)
    with open(study_id_csv_path, "r") as f:
        ids_csv = pd.read_csv(f)
        study_ids = ids_csv["study_id"]
        dicom_ids = ids_csv["dicom_id"]
        subject_ids = ids_csv["subject_id"]
    generated_reports["study_id"] = study_ids
    generated_reports["dicom_id"] = dicom_ids
    generated_reports["subject_id"] = subject_ids
    #generated_reports.drop_duplicates(subset=["study_id"], keep="first")
    generated_reports.to_csv(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, nargs='?', required=True,
                        help='path to model checkpoint')
    
    parser.add_argument('--data_path', type=str, nargs='?', required=False,
                        help='path to folder containing reports')
    
    parser.add_argument('--out_path', type=str, nargs='?', required=True,
                        help='path to file to write results')
    
    parser.add_argument('--cuda_device', type=int, nargs='?', required=False,
                        default = -1, help='id of GPU, if to use')

    
    args = parser.parse_args()
    
    run(args.model_path, args.data_path, args.out_path, args.cuda_device)
