"""Evaluates the model"""

import argparse
import logging
import os
import json
import torch
import imageio

import numpy as np

from utils import utils
from easydict import EasyDict
from utils.manager import Manager

import model.net as net
import model.data_loader as data_loader
from model.loss import compute_losses, compute_metrics
from termcolor import colored
from transform.transforms_lib import stn

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


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
    # print(manager.val_status)
    for k, v in manager.val_status.items():
        manager.val_status[k].reset()
        # manager.test_status[k].reset()

    # compute metrics over the dataset
    for data_batch in manager.val_dataloader:
        # move to GPU if available
        data_batch = utils.tensor_gpu(data_batch)
        # _image_concat = {"imgs": torch.cat([data_batch["img1"], data_batch["img2"]], 1)}
        data_batch["imgs"] = torch.cat([data_batch["img1"], data_batch["img2"]], 1)

        # compute model output
        output_batch = model(data_batch)

        # loss = compute_losses(_data_batch, output_batch, manager)

        # compute all metrics on this batch and auto update to manager
        compute_metrics(data_batch, output_batch, manager)

        # # 测试STN和输入的光流图
        # img1_warp = stn.dlt_spatial_transform(-1 * data_batch["gyro_field"].cuda(), data_batch["img1"].cuda())
        # img1_warp_np = np.uint8(np.transpose(img1_warp.cpu().detach().numpy().squeeze(), (1, 2, 0)))
        # img2 = np.uint8(np.transpose(data_batch["img2"].cpu().detach().numpy().squeeze(), (1, 2, 0)))
        # with imageio.get_writer('test.gif', mode='I', duration=0.5) as writer:
        #     writer.append_data(img1_warp_np)
        #     writer.append_data(img2)
        #
        # break

        # Update
        manager.train_status['cur_val_score'] = manager.val_status['epe'].avg
        # manager.test_status['cur_test_score'] = manager.test_status['epe']

    manager.writer.add_scalar("EPE/valid", manager.val_status['epe'].avg, manager.train_status['epoch'])
    print_metrics(manager)


def print_metrics(manager):
    print_str = ''
    for k, v in manager.val_status.items():
        print_str += '{}: {}  '.format(k, v.avg)
    manager.logger.info(colored('Val Results: ', 'red', attrs=['bold']))
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

    # Set the random seed for reproducible experiments
    # torch.manual_seed(230)
    # if params.cuda:
    #     torch.cuda.manual_seed(230)

    manager = Manager()
    manager.params = params
    manager.params.update(vars(args))

    manager.params.restore_file = args.restore_file

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))
    manager.logger = logger

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['valid'], manager)
    manager.val_dataloader = dataloaders['valid']

    logging.info("- done.")

    # Define the model
    if params.cuda:
        model = net.PWCLite(params).cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        model = net.PWCLite(params)

    manager.train_status['model'] = model
    manager.load_checkpoints()

    logging.info("Starting evaluation")

    # Evaluate
    test_metrics = evaluate(model, manager)

    # save_path = os.path.join(
    #     args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    # utils.save_dict_to_json(test_metrics, save_path)
