"""Peform hyperparemeters search"""

import argparse
import collections
import itertools
import os
import sys

from utils import utils
from experiment_dispatcher import dispatcher, tmux

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments', help='Directory containing params.json')
parser.add_argument('--id', default=1, type=int, help="Experiment id")


def launch_training_job(exp_dir, exp_name, session_name, param_pool_dict, params, start_id=0):
    # 自动划分tmux窗口
    tmux_ops = tmux.TmuxOps()
    # 自动组合超参和实验id
    task_manager = dispatcher.Enumerate_params_dict(task_thread=0, if_single_id_task=True, **param_pool_dict)

    num_jobs = len([v for v in itertools.product(*param_pool_dict.values())])
    exp_cmds = []

    for job_id in range(num_jobs):
        param_pool = task_manager.get_thread(ind=job_id)
        for hyper_params in param_pool:
            job_name = 'exp_{}'.format(job_id + start_id)
            for k in hyper_params.keys():
                params.dict[k] = hyper_params[k]

            params.dict['model_dir'] = os.path.join(exp_dir, exp_name, job_name)
            model_dir = params.dict['model_dir']

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            # Write parameters in json file
            json_path = os.path.join(model_dir, 'params.json')
            params.save(json_path)

            python = params.python_path

            # Launch training with this config
            if params.restore_file:
                cmd = 'rlaunch --cpu={} --memory={} --gpu={} --positive-tags=titanxp -- {} train.py --model_dir {} --restore_file {}'.format(
                    params.cpu, params.memory, params.gpu, python, model_dir, params.restore_file)
            else:
                cmd = 'rlaunch --cpu={} --memory={} --gpu={} --positive-tags=titanxp -- {} train.py --model_dir {}'.format(
                    params.cpu, params.memory, params.gpu, python, model_dir)

            exp_cmds.append(cmd)

    tmux_ops.run_task(exp_cmds, task_name=exp_name, session_name=session_name)


def experiment():
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')  # 这是一个base的参数设置文件
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # 首先，本脚本起一个索引的作用，通过id, name便可以轻易地找到对应的实验
    # 其次，通过自动化脚本，设置好要刷的参数可以自动新建tmux窗口，并在对应窗口中自动起实验
    # 除了要刷的超参外，其余网络的参数以及申请的资源数量都在params.json中设定

    # 比如下面的例子，通过id=1, name=lr，将在对应文件夹下(./{parent_dir})创建子文件夹存储日志，模型和结果
    # 并在ws2中新建4个窗口，最后自动rlaunch资源启动刷学习率的实验

    if args.id == 1:  # 该id用于判断是刷什么参数的实验
        # e.g. model and logs will be stored under 'experiment_lr'
        name = "SGF_ablation_study"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 0
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['model_name'] = ["UFlowSGFFuse", "UFlowSGFMap"]
    elif args.id == 2:  # 该id用于判断是刷什么参数的实验
        # e.g. model and logs will be stored under 'experiment_lr'
        name = "SGF_ablation_study_GOF_clean"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 0
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['model_name'] = ["UFlowSGFFuse", "UFlowSGFMap", "UFlowSGF", "UFlow"]
    elif args.id == 3:  # 该id用于判断是刷什么参数的实验
        # e.g. model and logs will be stored under 'experiment_lr'
        name = "SGF_ablation_study_GOF_clean"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 4
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['model_name'] = ["UFlowSGFFuse", "UFlowSGFMap", "UFlowSGF", "UFlow"]
    elif args.id == 4:  # 该id用于判断是刷什么参数的实验
        # e.g. model and logs will be stored under 'experiment_lr'
        name = "SGF_ablation_study_GOF_clean"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 8
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['with_spatial_transform'] = [True, False]
        param_pool_dict['loss'] = [{"l1": 0, "ssim": 0, "tenary": 1}, {"l1": 0.15, "ssim": 0.85, "tenary": 0}]
        param_pool_dict['model_name'] = ["UFlowSGFFuse"]
        param_pool_dict['restore_file'] = ["experiments/experiment_SGF_ablation_study_GOF_clean/exp_0/model_latest.pth"]
    elif args.id == 5:  # 该id用于判断是刷什么参数的实验
        # e.g. model and logs will be stored under 'experiment_lr'
        name = "SGF_ablation_study_GOF_clean"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 12
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['with_spatial_transform'] = [True, False]
        param_pool_dict['loss'] = [{"l1": 0, "ssim": 0, "tenary": 1}, {"l1": 0.15, "ssim": 0.85, "tenary": 0}]
        param_pool_dict['model_name'] = ["UFlowSGFMap"]
        param_pool_dict['restore_file'] = ["experiments/experiment_SGF_ablation_study_GOF_clean/exp_1/model_latest.pth"]
    elif args.id == 6:  # 该id用于判断是刷什么参数的实验
        # e.g. model and logs will be stored under 'experiment_lr'
        name = "SGF_ablation_study_GOF_clean"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 16
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['with_spatial_transform'] = [True, False]
        param_pool_dict['loss'] = [{"l1": 0, "ssim": 0, "tenary": 1}, {"l1": 0.15, "ssim": 0.85, "tenary": 0}]
        param_pool_dict['model_name'] = ["UFlowSGF"]
        param_pool_dict['restore_file'] = ["experiments/experiment_SGF_ablation_study_GOF_clean/exp_2/model_latest.pth"]
    elif args.id == 7:  # 该id用于判断是刷什么参数的实验
        # e.g. model and logs will be stored under 'experiment_lr'
        name = "SGF_ablation_study_GOF_clean"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 20
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['with_spatial_transform'] = [True, False]
        param_pool_dict['loss'] = [{"l1": 0, "ssim": 0, "tenary": 1}, {"l1": 0.15, "ssim": 0.85, "tenary": 0}]
        param_pool_dict['model_name'] = ["UFlow"]
        param_pool_dict['restore_file'] = ["experiments/experiment_SGF_ablation_study_GOF_clean/exp_3/model_latest.pth"]
    elif args.id == 8:  # 该id用于判断是刷什么参数的实验
        # 20211012-200数据-GOF
        # lr 1e-3 加入weather transform
        name = "SGF_ablation_study_GOF_clean_200cases"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 0
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['model_name'] = ["UFlowSGFFuse", "UFlowSGFMap", "UFlowSGF", "UFlow"]
    elif args.id == 9:  # 该id用于判断是刷什么参数的实验
        # 20211012-200数据-GOF
        # 去掉weather transform
        name = "SGF_ablation_study_GOF_clean_200cases"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 0
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['model_name'] = ["UFlowSGFFuse", "UFlowSGFMap", "UFlowSGF", "UFlow"]
        param_pool_dict['hyperparameters'] = [{
            "learning_rate": 1e-4,
            "batch_size": 4,
            "val_batch_size": 1
        }, {
            "learning_rate": 1e-5,
            "batch_size": 4,
            "val_batch_size": 1
        }]
    elif args.id == 10:  # 该id用于判断是刷什么参数的实验
        # 20211012-400数据-GOF_clean-GOF_final
        name = "SGF_ablation_study_GOF_clean_final_400_case"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 0
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['model_name'] = ["UFlowSGFFuse", "UFlowSGFMap", "UFlowSGF", "UFlow"]
        param_pool_dict['hyperparameters'] = [{"learning_rate": 1e-4, "batch_size": 4, "val_batch_size": 1}]
    elif args.id == 11:  # 该id用于判断是刷什么参数的实验
        # 20220415 gof-train有5个类，大概1w组数据
        name = "SGF_ablation_study_GOF_clean_final_400_case"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 0
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['model_name'] = ["UFlowSGFFuse", "UFlowSGFMap", "UFlowSGF", "UFlow"]
        param_pool_dict['hyperparameters'] = [{"learning_rate": 1e-4, "batch_size": 4, "val_batch_size": 1}]
    elif args.id == 12:  # 该id用于判断是刷什么参数的实验
        # 20220426 SGF做refine
        name = "SGF_refine"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 0
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['model_name'] = ["UFlowSGF"]
        param_pool_dict['with_spatial_transform'] = [True, False]
        param_pool_dict['hyperparameters'] = [{"learning_rate": 1e-5, "batch_size": 4, "val_batch_size": 1}]
        param_pool_dict['loss'] = [{"l1": 0, "ssim": 0, "tenary": 1}]
        param_pool_dict['restore_file'] = [
            "/data/fusion_gyro_optical_flow/deep_gyro_of_fusion/work/SGF_ablation_study/experiments/experiment_SGF_ablation_study_GOF_clean_final_400_case/exp_2/test_model_best.pth"
        ]
    elif args.id == 13:  # 该id用于判断是刷什么参数的实验
        # 20220426 UFlow做refine
        name = "UFlow_refine"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 0
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['model_name'] = ["UFlow"]
        param_pool_dict['with_spatial_transform'] = [True]
        param_pool_dict['hyperparameters'] = [{"learning_rate": 1e-5, "batch_size": 4, "val_batch_size": 1}]
        param_pool_dict['loss'] = [{"l1": 0, "ssim": 0, "tenary": 1}]
        param_pool_dict['restore_file'] = [
            "/data/fusion_gyro_optical_flow/deep_gyro_of_fusion/work/SGF_ablation_study/experiments/experiment_SGF_ablation_study_GOF_clean_final_400_case/exp_3/test_model_best.pth"
        ]
    elif args.id == 14:  # 该id用于判断是刷什么参数的实验
        # 20220429 SGF不用census loss nor ar transform
        name = "SGF_refine"
        session_name = 'exp'  # 会在名字为session_name的tmux session中开刷点实验, 需要提前建立
        start_id = 5
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['model_name'] = ["UFlowSGF"]
        param_pool_dict['hyperparameters'] = [{"learning_rate": 1e-5, "batch_size": 4, "val_batch_size": 1}]
        param_pool_dict['restore_file'] = [
            "/data/fusion_gyro_optical_flow/deep_gyro_of_fusion/work/SGF_ablation_study/experiments/experiment_SGF_ablation_study_GOF_clean_final_400_case/exp_2/test_model_best.pth"
        ]
    else:
        raise NotImplementedError

    launch_training_job(args.parent_dir, exp_name, session_name, param_pool_dict, params, start_id)


if __name__ == "__main__":
    experiment()
