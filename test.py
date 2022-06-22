"""Evaluates the model"""

import argparse
import logging
import os
import json
import torch

import torch.nn.functional as F

from easydict import EasyDict
from torchvision import transforms

import model.net as net
import model.data_loader as data_loader

from utils import utils
from utils.manager import Manager
from loss.loss import compute_metrics, compute_test_metrics
from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--gif_name')
parser.add_argument('--restore_file',
                    default='best',
                    help="name of the file in --model_dir \
                     containing weights to load")


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 64) + 1) * 64 - self.ht) % 64
        pad_wd = (((self.wd // 64) + 1) * 64 - self.wd) % 64
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def evaluate(model, manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    # set model to evaluation mode
    model.eval()

    # val/test status initial
    # print(manager.test_status)
    for k, v in manager.test_status.items():
        manager.test_status[k].reset()

    # compute metrics over the dataset
    for idx, loader in enumerate(manager.test_dataloader):
        # we only evaluate at gof-clean
        if idx != 0:
            continue
        for data_batch in loader:
            transformer = transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])
            # move to GPU if available
            data_batch = utils.tensor_gpu(data_batch)

            img1, img2 = data_batch['img1'], data_batch['img2']
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)

            data_batch["imgs"] = torch.cat([transformer(img1), transformer(img2)], 1)

            # compute model output
            output_batch = model(data_batch)

            output_batch["flow_fw"][0] = padder.unpad(output_batch["flow_fw"][0])

            # compute all metrics on this batch and auto update to manager
            compute_test_metrics(data_batch, output_batch, manager, "nn_epe")

            if data_batch["label"][0] == "RE":
                compute_test_metrics(data_batch, output_batch, manager, "ours_RE_None")
            elif data_batch["label"][0] == "Rain":
                compute_test_metrics(data_batch, output_batch, manager, "ours_Rain")
            elif data_batch["label"][0] == "Dark":
                compute_test_metrics(data_batch, output_batch, manager, "ours_LL")
            elif data_batch["label"][0] == "Fog":
                compute_test_metrics(data_batch, output_batch, manager, "ours_Fog")
            elif data_batch["label"][0] == "SNOW":
                compute_test_metrics(data_batch, output_batch, manager, "ours_SNOW")

            # Update
            # manager.train_status['cur_test_score'] = manager.test_status['ours_RE_None'].avg
            manager.train_status['cur_test_score'] = manager.test_status['nn_epe'].avg

    print_test_metrics(manager)


def test(model, manager, split_label=False):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    # set model to evaluation mode
    model.eval()

    # val/test status initial
    for k, v in manager.val_status.items():
        manager.val_status[k].reset()
    for k, v in manager.test_status.items():
        manager.test_status[k].reset()

    # image index
    idx = 0

    # compute metrics over the dataset
    for idx, loader in enumerate(manager.test_dataloader):
        for data_batch in loader:
            transformer = transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])
            # move to GPU if available
            data_batch = utils.tensor_gpu(data_batch)

            img1, img2 = data_batch['img1'], data_batch['img2']
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)

            data_batch["imgs"] = torch.cat([transformer(img1), transformer(img2)], 1)

            mask = None

            # compute model output
            output_batch = model(data_batch)
            output_batch["flow_fw"][0] = padder.unpad(output_batch["flow_fw"][0])

            # compute all metrics on this batch and auto update to manager
            gyro_batch = {"flow_fw": [data_batch["gyro_field"]]}

            identity_batch = {"flow_fw": [torch.zeros_like(data_batch["gyro_field"]).cuda()]}

            metrics_iden = compute_test_metrics(data_batch, identity_batch, manager, "identity_epe", mask)
            metrics_gyro = compute_test_metrics(data_batch, gyro_batch, manager, "gyro_epe", mask)
            metrics_nn = compute_test_metrics(data_batch, output_batch, manager, "nn_epe", mask)
            compute_metrics(data_batch, output_batch, manager, "nn_epe_" + str(idx))

            if split_label:
                # print(data_batch["label"][0])
                if data_batch["label"][0] == "Rain":
                    compute_test_metrics(data_batch, identity_batch, manager, "I33_Rain_None", mask)
                    compute_test_metrics(data_batch, gyro_batch, manager, "gyro_Rain_None", mask)
                    compute_test_metrics(data_batch, output_batch, manager, "ours_Rain_None", mask)
                    compute_metrics(data_batch, output_batch, manager, "ours_Rain_" + str(idx))
                elif data_batch["label"][0] == "RE":
                    compute_test_metrics(data_batch, identity_batch, manager, "I33_RE_None", mask)
                    compute_test_metrics(data_batch, gyro_batch, manager, "gyro_RE_None", mask)
                    compute_test_metrics(data_batch, output_batch, manager, "ours_RE_None", mask)
                    compute_metrics(data_batch, output_batch, manager, "ours_RE_" + str(idx))
                elif data_batch["label"][0] == "Dark":
                    compute_test_metrics(data_batch, identity_batch, manager, "I33_LL_None", mask)
                    compute_test_metrics(data_batch, gyro_batch, manager, "gyro_LL_None", mask)
                    compute_test_metrics(data_batch, output_batch, manager, "ours_LL_None", mask)
                    compute_metrics(data_batch, output_batch, manager, "ours_LL_" + str(idx))
                elif data_batch["label"][0] == "Fog":
                    compute_test_metrics(data_batch, identity_batch, manager, "I33_Fog_None", mask)
                    compute_test_metrics(data_batch, gyro_batch, manager, "gyro_Fog_None", mask)
                    compute_test_metrics(data_batch, output_batch, manager, "ours_Fog_None", mask)
                    compute_metrics(data_batch, output_batch, manager, "ours_Fog_" + str(idx))
                elif data_batch["label"][0] == "SNOW":
                    compute_test_metrics(data_batch, identity_batch, manager, "I33_SNOW_None", mask)
                    compute_test_metrics(data_batch, gyro_batch, manager, "gyro_SNOW_None", mask)
                    compute_test_metrics(data_batch, output_batch, manager, "ours_SNOW_None", mask)
                    compute_metrics(data_batch, output_batch, manager, "ours_SNOW_" + str(idx))
                else:
                    raise Exception("wrong scene type: {}".format(data_batch["label"][0]))

            print_metrics(manager)
            print_test_metrics(manager)

            # Update
            manager.train_status['cur_test_score'] = manager.test_status['nn_epe'].avg

    print_test_metrics(manager)


def print_metrics(manager):
    print_str = ''
    for k, v in manager.val_status.items():
        print_str += '{}: {:.4f}  '.format(k, v.avg)
    manager.logger.info(colored('Val Results: ', 'green', attrs=['bold']))
    manager.logger.info(colored(print_str, 'green', attrs=['bold']))
    # print('==========================')


def print_test_metrics(manager):
    print_str = ''
    for k, v in manager.test_status.items():
        print_str += '{}: {:.4f}  '.format(k, v.avg)
    manager.logger.info(colored('Test Results: ', 'red', attrs=['bold']))
    manager.logger.info(colored(print_str, 'red', attrs=['bold']))
    # print('==========================')


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    with open(json_path) as f:
        params = EasyDict(json.load(f))

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    manager = Manager()
    manager.params = params
    manager.params.update(vars(args))

    manager.params.restore_file = args.restore_file
    manager.gif_name = args.gif_name

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))
    manager.logger = logger

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], manager)
    manager.test_dataloader = dataloaders['test']

    logging.info("- done.")

    # Define the model and optimizer
    if params.model_name == "GyroFlow":
        model = net.GyroFlow(params)
    else:
        raise NotImplementedError

    # Define the model
    if params.cuda:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    manager.train_status['model'] = model

    manager.load_checkpoints()

    logging.info("Starting evaluation")

    # Evaluate
    test_metrics = test(model, manager, split_label=True)
