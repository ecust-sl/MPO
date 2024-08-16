import copy
import logging
import os
import time
from abc import abstractmethod
from modules.metrics import compute_mlc
import pandas as pd
import torch
from numpy import inf
import itertools
import wandb
from modules.optimizers import set_lr,get_lr
from modules.rewards import get_self_critical_reward, init_scorer, sentence_score, corpus_score
from modules.loss import compute_loss
import wandb
import numpy as np
import sys
from .pvf_coder import update_prefer_vector

def generate_permutations(interval=0.05):
    values = np.arange(0, 1.01, 0.01)
    combinations = list(itertools.product(values, repeat=2))

    filtered_combinations = [comb for comb in combinations if np.isclose(sum(comb), 1.0)]

    return filtered_combinations
class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, ve_optimizer, ed_optimizer, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        # print('11111111111111111111111111=',os.environ['CUDA_VISIBLE_DEVICES'])
        print('devices =====' ,self.device)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.ve_optimizer = ve_optimizer
        self.ed_optimizer = ed_optimizer
        self.chexbert_path = '/home/shilei/project/R2GenRL/CheXbert/checkpoint/chexbert.pth'

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        self.mnt_best_all = inf if self.mnt_mode == 'min' else -inf


        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.preference_vectors_three_dimensions = [[0,1],[0.1,0.9],[0.2,0.8],[0.3,0.7],[0.4,0.6],[0.5,0.5],
                                                    [0.6,0.4],[0.7,0.3],[0.8,0.2],[0.9,0.1],[1,0]]

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        self.index = 0

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        # wandb.init(project = 'R2GenRL-Pan-new', name = 'IU_0609_Paneca_B4_RG_11')
        for epoch in range(self.start_epoch, self.epochs + 1):

            result = self._train_epoch(epoch)
            # set_lr(self.ve_optimizer, 0.8 * self.ve_optimizer.current_lr)
            # set_lr(self.ed_optimizer, 0.8 * self.ed_optimizer.current_lr)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)
            self._print_to_file(log)
            best = False

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    # improved = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.mnt_best) or \
                    #            (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.mnt_best)
                    if self.mnt_mode == 'max':
                        # if log['test_ROUGE_L'] + log['test_BLEU_1'] + log['test_BLEU_4'] >= self.mnt_best_all:
                        #     improved = True
                        # else:
                        #     improved = False
                        if log['test_ROUGE_L'] + log['test_BLEU_1'] + log['test_BLEU_4']  >= self.mnt_best_all:
                            improved = True
                        else:
                            improved = False
                    elif self.mnt_mode == 'min':
                        if log['test_ROUGE_L'] + log['test_BLEU_1'] + log['test_BLEU_4'] <= self.mnt_best_all:
                            improved = True
                        else:
                            improved = False
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric_test]
                    self.mnt_best_all = log['test_ROUGE_L'] + log['test_BLEU_1'] + log['test_BLEU_4']
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_best(self, epoch, log):
        # evaluate model performance according to configured metric, save best checkpoint as model_best
        best = False
        if self.mnt_mode != 'off':
            try:
                # check whether model performance improved or not, according to specified metric(mnt_metric)
                # improved = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.mnt_best) or \
                #            (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.mnt_best)
                if self.mnt_mode == 'max':
                    # if log['test_ROUGE_L'] + log['test_BLEU_1'] + log['test_BLEU_4'] >= self.mnt_best_all:
                    #     improved = True
                    # else:
                    #     improved = False
                    if log['test_ROUGE_L'] + log['test_BLEU_1'] + log['test_BLEU_4']  >= self.mnt_best_all:
                        improved = True
                    else:
                        improved = False
                elif self.mnt_mode == 'min':
                    if log['test_ROUGE_L'] + log['test_BLEU_1'] + log['test_BLEU_4'] <= self.mnt_best_all:
                        improved = True
                    else:
                        improved = False
            except KeyError:
                self.logger.warning(
                    "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                self.mnt_mode = 'off'
                improved = False

            if improved:
                self.mnt_best = log[self.mnt_metric_test]
                # self.mnt_best_all = log['test_ROUGE_L'] + log['test_BLEU_1'] + log['test_BLEU_4']
                self.mnt_best_all = log['test_ROUGE_L'] + log['test_BLEU_1'] + log['test_BLEU_4']
                best = True

            self._save_checkpoint(epoch, save_best=best)

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_to_file(self, log):
        crt_time = time.asctime(time.localtime(time.time()))
        log['time'] = crt_time
        log['seed'] = self.args.seed
        log['best_model_from'] = 'train'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir,
                                   self.args.dataset_name + '_rl' + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        tmp_log = copy.deepcopy(log)
        tmp_log.update(**self.args.__dict__)
        record_table = pd.concat([record_table, pd.DataFrame([tmp_log])], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _get_learning_rate(self):
        lrs = list()
        lrs.append(self.ve_optimizer.current_lr)
        lrs.append(self.ed_optimizer.current_lr)

        return {'lr_visual_extractor': lrs[0], 'lr_encoder_decoder': lrs[1]}

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        # print('n_gpu == ' , n_gpu)
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:1' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            've_optimizer': self.ve_optimizer.state_dict(),
            'ed_optimizer': self.ed_optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location = "cuda:1")
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best_all = 0.475 + 0.18 + 0.375

        self.model.load_state_dict(checkpoint['state_dict'], strict = False)
        # self.ve_optimizer.load_state_dict(checkpoint['ve_optimizer'])
        # self.ed_optimizer.load_state_dict(checkpoint['ed_optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _write_to_file(self, gts, res, epoch, iter):
        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        fgt = open(os.path.join(self.args.record_dir, 'gts-{}-{}.txt'.format(epoch, iter)), 'w')
        for gt in gts:
            fgt.write(gt + '\n')
        fre = open(os.path.join(self.args.record_dir, 'res-{}-{}.txt'.format(epoch, iter)), 'w')
        for re in res:
            fre.write(re + '\n')


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, ve_optimizer, ed_optimizer, args, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, ve_optimizer, ed_optimizer, args)
        # self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _set_lr_ve(self):
        # if iteration < self.args.noamopt_warmup:
        #     current_lr = self.args.lr_ve * (iteration + 1) / self.args.noamopt_warmup
        #     set_lr(self.ve_optimizer, current_lr)
        current_lr_ve = get_lr(self.ve_optimizer)
        current_lr_ve = max(current_lr_ve * 0.8,1e-5)
        set_lr(self.ve_optimizer, current_lr_ve)




    def _set_lr_ed(self):
        # if iteration < self.args.noamopt_warmup:
        #     current_lr = self.args.lr_ed * (iteration + 1) / self.args.noamopt_warmup
        #     set_lr(self.ed_optimizer, current_lr)
        current_lr_ed = get_lr(self.ed_optimizer)
        current_lr_ed = max(current_lr_ed * 0.8,2e-5)
        set_lr(self.ed_optimizer, current_lr_ed)

    def _set_lr_ve_it(self, iteration):
        # if iteration < self.args.noamopt_warmup:
        #     current_lr = self.args.lr_ve * (iteration + 1) / self.args.noamopt_warmup
        #     set_lr(self.ve_optimizer, current_lr)
        current_lr_ed = get_lr(self.ed_optimizer)
        current_lr_ve = current_lr_ed * 0.1
        set_lr(self.ve_optimizer, current_lr_ve)

    def _set_lr_ed_it(self, iteration):
        if iteration < self.args.noamopt_warmup:
            current_lr = self.args.lr_ve * (iteration + 1) / self.args.noamopt_warmup
            set_lr(self.ed_optimizer, current_lr)
        # current_lr_ed = get_lr(self.ed_optimizer)
        # current_lr_ve = current_lr_ed * 0.1
        # set_lr(self.ve_optimizer, current_lr_ve)


    def _train_epoch(self, epoch):

        # hyp = self.preference_vectors_three_dimensions[self.index]
        hyp = self.preference_vectors_three_dimensions[self.index]
        print('b4:', hyp[0], 'rg:', hyp[1])
        self.index = (self.index + 1) % len(self.preference_vectors_three_dimensions)

        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        train_loss_rl = 0
        train_loss_nll = 0
        sum_reward = 0
        self.model.train()
        # if epoch > 40:
        #     self._set_lr_ed()
        #     self._set_lr_ve()

        # print('start ================ ')



        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            # update_prefer_vector(self.preference_vectors_three_dimensions[self.index])
            update_prefer_vector(hyp)
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device)

            # ********* Self-Critical *********
            # iteration = batch_idx + (epoch - 1) * len(self.train_dataloader)
            init_scorer()





            # if epoch <= 40:
            #     loss = loss_nll
            # else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(images, mode='sample',
                                           update_opts={'sample_method': self.args.sc_sample_method,
                                                        'beam_size': self.args.sc_beam_size})

            self.model.train()
            gen_result, sample_logprobs = self.model(images, mode='sample',
                                                     update_opts={'sample_method': self.args.train_sample_method,
                                                                  'beam_size': self.args.train_beam_size,
                                                                  'sample_n': self.args.train_sample_n})


            generated_tokens = gen_result.squeeze().tolist()
            # print('tokens ================', generated_tokens)
            # print('sample_logprobs ====', sample_logprobs[0][0][684])
            # print('probs =====', logprobs[0][0][684])
            gts = reports_ids[:, 1:]
            reward1, reward2 = get_self_critical_reward(greedy_res, gts, gen_result, hyp, self.model.tokenizer, images_id)



            reward1, reward2 = torch.from_numpy(reward1).to(sample_logprobs), torch.from_numpy(reward2).to(sample_logprobs)
            # reward_val = torch.sum(reward) / len(reward.flatten())
            loss_rl_1 = self.criterion(sample_logprobs, gen_result.data, reward1)
            loss_rl_2 = self.criterion(sample_logprobs, gen_result.data, reward2)

            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                reports_masks.to(self.device)

            output = self.model(images, reports_ids, mode='train')

            # reports_t = self.model.tokenizer.decode_batch(output.cpu().numpy())
            # ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            # print('report_gen:', reports_t)
            # print('report_ori:', ground_truths)

            loss_nll = compute_loss(output, reports_ids, reports_masks)

            loss = 0.01 * loss_nll + 0.99 * (hyp[0] * loss_rl_1 + hyp[1] * loss_rl_2)

            train_loss += loss.item()
            # if epoch > 40:
            train_loss_rl += 0.99 * (hyp[0] * loss_rl_1 + hyp[1] * loss_rl_2).item()
            train_loss_nll += loss_nll.item()

            # print('final================',self.device)
            self.ve_optimizer.zero_grad()
            self.ed_optimizer.zero_grad()

            loss.backward()
            self.ve_optimizer.step()
            self.ed_optimizer.step()




            # self.ve_optimizer.zero_grad()
            # self.ed_optimizer.zero_grad()
            # self.ve_optimizer.step()
            # self.ed_optimizer.step()
            if batch_idx % self.args.log_period == 0:
                lrs = self._get_learning_rate()
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.6f}, LR (ve): {:.6f}, LR (ed): {:6f}.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         train_loss / (batch_idx + 1), lrs['lr_visual_extractor'],
                                         lrs['lr_encoder_decoder']))

            if (batch_idx+1) % self.args.sc_eval_period == 0:
                log = {'train_loss': train_loss / (batch_idx + 1)}
                # self.index += 1


                self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
                val_loss = 0
                self.model.eval()
                with torch.no_grad():
                    # val_loss = 0
                    val_gts, val_res = [], []
                    for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                        images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                            self.device), reports_masks.to(self.device)

                        # # ****** Compute Loss ******
                        # images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                        #                                      reports_masks.to(self.device)
                        # output = self.model(images, reports_ids, mode='train')
                        # loss = self.criterion(output, reports_ids, reports_masks)
                        # val_loss += loss.item()
                        # # ****** Compute Loss ******

                        output, _ = self.model(images, mode='sample')
                        # loss_nll = compute_loss(output, reports_ids, reports_masks)

                        reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                        ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                        val_res.extend(reports)
                        val_gts.extend(ground_truths)


                        # for id, re, gt in zip(images_id, reports, ground_truths):
                        #     print(id)
                        #     print('[Generated]: {}'.format(re))
                        #     print('[Ground Truth]: {}'.format(gt))
                    # gen_label = label(self.chexbert_path, val_res)  
                    # gt_label = label(self.chexbert_path, val_gts)
                    # gt_label = np.array(gt_label)
                    # gen_label = np.array(gen_label)
                    # _, gen_ce = compute_mlc(gt_label, gen_label)
                    val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                               {i: [re] for i, re in enumerate(val_res)})

                    log.update(**{'val_' + k: v for k, v in val_met.items()})
                    # log.update({'val_macro_f1': gen_ce})
                    # val_gts_move = []
                    # val_gts_move.append(val_gts)
                    # val_move_score = corpus_score(val_res, val_gts_move)
                    # log.update(({'val_move': np.mean(val_move_score)}))
                    # for k, v in val_met.items():
                    #     if k == 'BLEU_1':
                    #         print('val_bleu_1:', v)
                    #     if k == 'BLEU_2':
                    #         print('val_bleu_2:', v)
                    #     if k == 'BLEU_3':
                    #         print('val_bleu_3:', v)
                    #     if k == 'BLEU_4':
                    #         print('val_bleu_4:', v)
                    #     if k == 'METEOR':
                    #         print('val_meteor:', v)
                    #     if k == 'ROUGE_L':
                    #         print('val_ROUGE_L:', v)

                self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
                self.model.eval()
                with torch.no_grad():
                    test_gts, test_res = [], []
                    for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                        images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                            self.device), reports_masks.to(self.device)
                        output, _ = self.model(images, mode='sample')
                        reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                        ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                        test_res.extend(reports)
                        test_gts.extend(ground_truths)

                        # for id, re, gt in zip(images_id, reports, ground_truths):
                        #     print(id)
                        #     print('[Generated]: {}'.format(re))
                        #     print('[Ground Truth]: {}'.format(gt))
                    # gen_label = label(self.chexbert_path, test_res)  
                    # gt_label = label(self.chexbert_path, test_gts)
                    # gt_label = np.array(gt_label)
                    # gen_label = np.array(gen_label)
                    # _, gen_ce = compute_mlc(gt_label, gen_label)
                    test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                                {i: [re] for i, re in enumerate(test_res)})
                    log.update(**{'test_' + k: v for k, v in test_met.items()})
                    # log.update({'test_macro_f1': gen_ce})
                    # test_gts_move = []
                    # test_gts_move.append(val_gts)
                    # test_move_score = corpus_score(test_res, test_gts_move)
                    # log.update(({'test_move': np.mean(test_move_score)}))
                    # for k, v in test_met.items():
                    #     if k == 'BLEU_1':
                    #         print('test_bleu_1:', v)
                    #     if k == 'BLEU_2':
                    #         print('test_bleu_2:', v)
                    #     if k == 'BLEU_3':
                    #         print('test_bleu_3:', v)
                    #     if k == 'BLEU_4':
                    #         print('test_bleu_4:', v)
                    #     if k == 'METEOR':
                    #         print('test_meteor:', v)
                    #     if k == 'ROUGE_L':
                    #         print('test_ROUGE_L:', v)
                self._save_best(epoch, log)
                self._print_to_file(log)
                self._write_to_file(test_gts, test_res, epoch, batch_idx)

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        # wandb.log({'train_loss': train_loss / (batch_idx + 1), 'train_loss_rl': train_loss_rl / (batch_idx + 1),
        #            'train_loss_nll': train_loss_nll / (batch_idx + 1)})


        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            # val_loss = 0
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                # # ****** Compute Loss ******
                # images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                #                                      reports_masks.to(self.device)
                # output = self.model(images, reports_ids, mode='train')
                # loss = self.criterion(output, reports_ids, reports_masks)
                # val_loss += loss.item()
                # # ****** Compute Loss ******

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

                # for id, re, gt in zip(images_id, reports, ground_truths):
                #     print(id)
                #     print('[Generated]: {}'.format(re))
                #     print('[Ground Truth]: {}'.format(gt))
            # gen_label = label(self.chexbert_path, val_res)  
            # gt_label = label(self.chexbert_path, val_gts)
            # gt_label = np.array(gt_label)
            # gen_label = np.array(gen_label)
            # _, gen_ce = compute_mlc(gt_label, gen_label)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            # val_entity_f1, val_relation_f1 = cal_rad(val_gts, val_res)
            # log.update({'val_entity_f1': np.mean(val_entity_f1), 'val_relation_f1': np.mean(val_relation_f1)})
            # val_gts_move = []
            # val_gts_move.append(val_gts)
            # val_move_score = corpus_score(val_res, val_gts_move)
            # log.update(({'val_move': np.mean(val_move_score)}))

            b4 = 0
            # meteor_v = 0
            b1 = 0
            rg = 0
            for k, v in val_met.items():
                if k == 'BLEU_4':
                    b4 = v
                if k == 'BLEU_1':
                    b1 = v
                if k =='ROUGE_L':
                    rg = v


            # wandb.log({'val_bleu_4': b4})
            # wandb.log({'val_bleu_1': b1})
            # wandb.log({'val_rouge': rg})
            # log.update(**{'val_loss': val_loss / len(self.val_dataloader)})

        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

                # for id, re, gt in zip(images_id, reports, ground_truths):
                #     print(id)
                #     print('[Generated]: {}'.format(re))
                #     print('[Ground Truth]: {}'.format(gt))
            # gen_label = label(self.chexbert_path, test_res)  
            # gt_label = label(self.chexbert_path, test_gts)
            # gt_label = np.array(gt_label)
            # gen_label = np.array(gen_label)
            # _, gen_ce = compute_mlc(gt_label, gen_label)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            # log.update({'test_macro_f1': gen_ce})
            # test_gts_move = []
            # test_gts_move.append(val_gts)
            # test_move_score = corpus_score(test_res, test_gts_move)
            # log.update(({'test_move': np.mean(test_move_score)}))
            # test_entity_f1, test_relation_f1 = cal_rad(test_gts, test_res)
            # log.update({'test_entity_f1': np.mean(test_entity_f1), 'test_relation_f1': np.mean(test_relation_f1)})
            b4_t = 0
            # meteor_v = 0
            b1_t = 0
            rg_t = 0
            for k, v in test_met.items():
                if k == 'BLEU_4':
                    b4_t = v
                if k == 'BLEU_1':
                    b1_t = v
                if k == 'ROUGE_L':
                    rg_t = v

            # wandb.log({'test_bleu_4': b4_t})
            # wandb.log({'test_bleu_1': b1_t})
            # wandb.log({'test_rg': rg_t})
        log.update(**self._get_learning_rate())
        # self.ed_optimizer.current_lr = self.ed_optimizer.current_lr * 0.8
        # self.ve_optimizer.current_lr = self.ve_optimizer.current_lr * 0.8
        self._write_to_file(test_gts, test_res, epoch, 0)
        # print('lr_ve:', get_lr(self.ve_optimizer))
        # print('lr_ed:', get_lr(self.ed_optimizer))
        # self.ed_optimizer.current_lr = get_lr(self.ed_optimizer)
        # self.ve_optimizer.current_lr = get_lr(self.ve_optimizer)
        # if epoch == 40:
        #     set_lr(self.ve_optimizer, 5e-5)
        #     set_lr(self.ed_optimizer, 1e-4)

        # self.lr_scheduler.step()


        return log
