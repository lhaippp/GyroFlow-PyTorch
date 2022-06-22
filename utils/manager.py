import json
import logging
import os
import shutil
import time

import torch
import pickle
import boto3
from utils import utils
from collections import defaultdict
import numpy as np


class Manager():
    def __init__(self):
        # logger
        self.logger = None

        # data status
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        # train status
        self.train_status = {}
        self.train_status['epoch'] = 0
        self.train_status['step'] = 0
        self.train_status['model'] = None
        self.train_status['optimizer'] = None
        self.train_status['scheduler'] = None
        self.train_status['print_str'] = ''
        self.train_status['best_val_score'] = 100
        self.train_status['best_test_score'] = 100
        self.train_status['cur_val_score'] = 100
        self.train_status['cur_test_score'] = 100

        # val status
        self.val_status = defaultdict(utils.AverageMeter)

        # test status
        self.test_status = defaultdict(utils.AverageMeter)

        # model status
        self.loss_status = defaultdict(utils.AverageMeter)

        # params status
        self.params = None

        # client init
        self.s3_client = boto3.client('s3', endpoint_url='http://oss.i.brainpp.cn')
        self.bucket_name = 'xhdata'

    def save_checkpoints(self):

        cur_val_is_best, cur_test_is_best = False, False
        if self.train_status['cur_val_score'] < self.train_status['best_val_score']:
            cur_val_is_best = True
            self.train_status['best_val_score'] = self.train_status['cur_val_score']

        if self.train_status['cur_test_score'] < self.train_status['best_test_score']:
            cur_test_is_best = True
            self.train_status['best_test_score'] = self.train_status['cur_test_score']

        state = {'state_dict': self.train_status['model'].state_dict(),
                 'optimizer': self.train_status['optimizer'].state_dict(),
                 'scheduler': self.train_status['scheduler'].state_dict(),
                 'step': self.train_status['step'],
                 'epoch': self.train_status['epoch'],
                 'best_val_score': self.train_status['best_val_score'],
                 'best_test_score': self.train_status['best_test_score']}

        # print("test")

        # for latest
        if self.train_status['epoch'] % 10 == 0:
            latest_save_name = os.path.join(self.params.model_dir, 'model_latest.pth')
            if self.params.save_mode == 'local':
                torch.save(state, latest_save_name)
            elif self.params.save_mode == 'oss':
                save_dict = pickle.dumps(state)
                resp = self.s3_client.put_object(Bucket=self.bucket_name, Key=latest_save_name, Body=save_dict[0:])
            else:
                raise NotImplementedError

            val_latest_metrics_name = os.path.join(self.params.model_dir, 'val_metrics_latest.json')
            test_latest_metrics_name = os.path.join(self.params.model_dir, 'test_metrics_latest.json')
            utils.save_dict_to_json(self.val_status, self.train_status['step'], self.train_status['epoch'], val_latest_metrics_name)
            utils.save_dict_to_json(self.test_status, self.train_status['step'], self.train_status['epoch'], test_latest_metrics_name)
            self.logger.info('Saved checkpoint to {}: {}'.format(self.params.model_dir, latest_save_name))

        # for val best
        if cur_val_is_best:
            best_save_name = os.path.join(self.params.model_dir, 'val_model_best.pth')
            best_metrics_name = os.path.join(self.params.model_dir, 'val_metrics_best.json')
            torch.save(state, best_save_name)
            utils.save_dict_to_json(self.val_status, self.train_status['step'], self.train_status['epoch'], best_metrics_name)
            self.logger.info(
                'Checkpoint is current val best, score={:.7f}'.format(self.val_status['epe'].avg))

        # for test best
        if cur_test_is_best:
            best_save_name = os.path.join(self.params.model_dir, 'test_model_best.pth')
            best_metrics_name = os.path.join(self.params.model_dir, 'test_metrics_best.json')
            torch.save(state, best_save_name)
            utils.save_dict_to_json(self.test_status, self.train_status['step'], self.train_status['epoch'], best_metrics_name)
            self.logger.info('Checkpoint is current test best, score={:.7f}'.format(self.test_status['nn_epe'].avg))

    def load_checkpoints(self):
        if self.params.save_mode == 'local':
            state = torch.load(self.params.restore_file)
        elif self.params.save_mode == 'oss':
            resp = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.params.restore_file[0:])
            state = resp['Body'].read()
            state = pickle.loads(state)
        else:
            raise NotImplementedError

        if not self.params.only_weights:
            # print(state)
            if 'step' in state:
                self.train_status['step'] = state['step']

            if 'epoch' in state:
                self.train_status['epoch'] = state['epoch'] + 1

            if 'best_val_status' in state:
                self.val_status['best_val_status'] = state['best_val_status']

            # if 'best_test_score' in state:
            #     self.test_status['best_test_score'] = state['best_test_score']

            if 'optimizer' in state and self.train_status['optimizer'] is not None:
                try:
                    self.train_status['optimizer'].load_state_dict(state['optimizer'])

                except:
                    optimizer_dict = self.train_status['optimizer'].state_dict()
                    state_dict = {k: v for k, v in state['optimizer'].items() if k in optimizer_dict.keys()}
                    optimizer_dict.update(state_dict)
                    self.train_status['optimizer'].load_state_dict(optimizer_dict)

            if 'scheduler' in state and self.train_status['scheduler'] is not None:
                try:
                    self.train_status['scheduler'].load_state_dict(state['scheduler'])

                except:
                    scheduler_dict = self.train_status['scheduler'].state_dict()
                    state_dict = {k: v for k, v in state['scheduler'].items() if k in scheduler_dict.keys()}
                    scheduler_dict.update(state_dict)
                    self.train_status['scheduler'].load_state_dict(scheduler_dict)

            self.logger.info('Loaded model from {}'.format(self.params.restore_file))

        if 'state_dict' in state and self.train_status['model'] is not None:
            try:
                self.train_status['model'].load_state_dict(state['state_dict'])
            except:
                net_dict = self.train_status['model'].state_dict()
                if 'module' not in list(state['state_dict'].keys())[0]:
                    state_dict = {'module.' + k: v for k, v in state['state_dict'].items() if
                                  'module.' + k in net_dict.keys()}
                else:
                    state_dict = {k: v for k, v in state['state_dict'].items() if k in net_dict.keys()}
                net_dict.update(state_dict)
                self.train_status['model'].load_state_dict(net_dict, strict=False)

        self.logger.info('Loaded models from {}'.format(self.params.restore_file))
