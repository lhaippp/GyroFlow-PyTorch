"""Train the model"""

import os
import json
import test
import torch
import argparse
import datetime

import torch.optim as optim

from tqdm import tqdm
from easydict import EasyDict

from utils import utils
from utils.manager import Manager
from loss.loss import compute_losses
from loss.loss import fetch_loss
from transform.ar_transforms.sp_transfroms import RandomAffineFlow

import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file',
                    default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('-ow', '--only_weights', default=False, action='store_true', help='Only use weights to load or load all train status.')


def train(model, manager):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # loss status
    for k, v in manager.loss_status.items():
        manager.loss_status[k].reset()

    # set model to training mode
    torch.cuda.empty_cache()
    model.train()

    # Use tqdm for progress bar
    with tqdm(total=len(manager.train_dataloader)) as t:
        for i, data_batch in enumerate(manager.train_dataloader):
            # move to GPU if available
            data_batch = utils.tensor_gpu(data_batch)
            data_batch["imgs"] = torch.cat([data_batch["img1"], data_batch["img2"]], 1)

            # infor print
            manager.train_status['print_str'] = 'Epoch: {:3d}, lr={:.6f}, spatial transform: {}'.format(
                manager.train_status['epoch'], manager.train_status['scheduler'].get_lr()[0], manager.params.with_spatial_transform)

            # compute model output and loss
            output_batch = model(data_batch)
            loss = compute_losses(data_batch, output_batch, manager)

            # clear previous gradients, compute gradients of all variables wrt loss
            manager.train_status['optimizer'].zero_grad()
            loss['total'].backward()

            # performs updates using calculated gradients
            manager.train_status['optimizer'].step()
            manager.train_status['step'] += 1

            t.set_description(desc=manager.train_status['print_str'])
            t.update()

    manager.train_status['scheduler'].step()
    manager.train_status['epoch'] += 1


def train_and_evaluate(model, manager):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
    """

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        manager.load_checkpoints()
        manager.params.fine_tune = True
    else:
        manager.params.fine_tune = False

    for epoch in range(manager.params.num_epochs):
        # compute number of batches in one epoch (one full pass over the training set)
        train(model, manager)

        # Test for one epoch on test set
        test.evaluate(model, manager)

        # Save weights
        manager.save_checkpoints()


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    # params = utils.Params(json_path)
    with open(json_path) as f:
        params = EasyDict(json.load(f))

    # initial status
    manager = Manager()
    manager.params = params
    manager.params.update(vars(args))

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Add ExpHub
    manager.session = None

    # Tensorboard writer
    manager.writer = None

    # Set the logger
    logger = utils.set_logger(os.path.join(manager.params.model_dir, 'train.log'))
    manager.logger = logger

    # Set the tensorboard writer
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create the input data pipeline
    logger.info("Loading the datasets from {}".format(manager.params.data_dir))

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'test'], manager)
    manager.train_dataloader = dataloaders['train']
    manager.test_dataloader = dataloaders['test']

    # Define the model and optimizer
    if params.model_name == "UFlowSGF":
        model = net.UFlowSGF(params)
    else:
        raise NotImplementedError

    # define loss function
    if manager.params.loss_type == "UnFlowLoss":
        unFlowLoss = fetch_loss(manager.params)
        manager.unFlowLoss = unFlowLoss
    else:
        raise NotImplementedError

    if params.cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # add regulizer to weights and bias
    param_groups = [
        {
            "params": utils.bias_parameters(model)
        },
        {
            "params": utils.weight_parameters(model),
            "weight_decay": 1e-6
        },
    ]

    manager.train_status['model'] = model
    manager.train_status['optimizer'] = optim.Adam(param_groups,
                                                   lr=manager.params.hyperparameters.learning_rate,
                                                   betas=(0.9, 0.999),
                                                   eps=1e-7)
    manager.train_status['scheduler'] = optim.lr_scheduler.ExponentialLR(manager.train_status['optimizer'], gamma=1)

    # Train the model
    logger.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, manager)
