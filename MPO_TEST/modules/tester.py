import logging
import os
import random
from abc import abstractmethod

import cv2
import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm
from modules.metrics import compute_mlc
from modules.utils import generate_heatmap
import sys

class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        self.chexbert = "/home/shilei/project/R2GenRL/CheXbert/checkpoint/chexbert.pth"

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        print('device == ' ,self.device)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._load_checkpoint(args.load)
        self.model = model.to(self.device)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path, map_location = torch.device("cuda:0"))
        # checkpoint.to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict = False)


class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        self.model.eval()
        log = dict()
        with torch.no_grad():
            test_gts, test_res = [], []
            test_gt_label, test_res_label = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, _ = self.model(images, mode='sample')
                # print('images == ' ,images)
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                # res_label = label(self.chexbert, reports)
                # gt_label = label(self.chexbert, ground_truths)
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                # for i in range(len(res_label)):
                #     # print('i========================', res_label[i])
                #     test_res_label.append(res_label[i])
                #     test_gt_label.append(gt_label[i])

            # print('label ================================', test_res_label)
            # test_res_label = np.array(test_res_label)
            # test_gt_label = np.array(test_gt_label)
            # test_res_label, test_gt_label = label_test(self.chexbert, test_res, test_gts)
            # _, _, _, test_met_ce = compute_mlc(test_gt_label, test_res_label)
            # log.update(**{'test_' + k: v for k, v in test_met_ce.items()})
            # test_met_ce = compute_mlc(test_gt_label, test_res_label)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            # test_met_ce = compute_mlc(test_gt_label, test_res_label)

            # test_entity_f1, test_relation_f1 = cal_rad(test_gts, test_res)

            log.update(**{'test_' + k: v for k, v in test_met.items()})
            # log.update(**{'test_' + k: v for k, v in test_met_ce.items()})
            # log.update({'test_entity_f1': np.mean(test_entity_f1), "test_relation_f1": np.mean(test_relation_f1)})
            print(log)
            # random_id = random.randint(0, len(test_res))
            #
            # print('test_report', test_res[random_id])
            # print('ge_report', test_gts[random_id])

            test_res, test_gts = pd.DataFrame(test_res), pd.DataFrame(test_gts)
            test_res.to_csv(os.path.join(self.save_dir, "res.csv"), index=False, header=False)
            test_gts.to_csv(os.path.join(self.save_dir, "gts.csv"), index=False, header=False)

        return log

    def plot(self):
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "attentions_entities"), exist_ok=True)
        ner = spacy.load("en_core_sci_sm")
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, _ = self.model(images, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()

                char2word = [idx for word_idx, word in enumerate(report) for idx in [word_idx] * (len(word) + 1)][:-1]

                attention_weights = self.model.encoder_decoder.attention_weights[:-1]
                assert len(attention_weights) == len(report)
                for word_idx, (attns, word) in enumerate(zip(attention_weights, report)):
                    for layer_idx, attn in enumerate(attns):
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)

                        heatmap = generate_heatmap(image, attn.mean(1).squeeze())
                        cv2.imwrite(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
                                    heatmap)

                for ne_idx, ne in enumerate(ner(" ".join(report)).ents):
                    for layer_idx in range(len(attention_weights[0])):
                        os.makedirs(os.path.join(self.save_dir, "attentions_entities", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)
                        attn = [attns[layer_idx] for attns in
                                attention_weights[char2word[ne.start_char]:char2word[ne.end_char] + 1]]
                        attn = np.concatenate(attn, axis=2)
                        heatmap = generate_heatmap(image, attn.mean(1).mean(1).squeeze())
                        cv2.imwrite(os.path.join(self.save_dir, "attentions_entities", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(ne_idx, ne)),
                                    heatmap)
