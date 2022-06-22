#!/usr/bin/env python3
import getpass
import functools
import torch
import git
import refile
import datetime

from io import BytesIO
from pathlib import Path
from typing import Tuple
from loguru import logger


from brainpp.exphub.watcher import ExpWatcher


def get_latest_commit(exp_dir) -> Tuple[str, int]:
    excludes = {"train_log", '__pycache__'}
    dep_paths = [
        str(x) for x in exp_dir.iterdir()
        if x.name not in excludes
    ]
    # dep_paths.append(str(runfile))
    repo = git.Repo()
    _git = repo.git
    rev_list = _git.rev_list('HEAD', "--max-count=1", '--', *dep_paths).split()
    assert len(rev_list) > 0
    head = rev_list[0]
    return head, repo.commit(head).count()


class Session:
    repo = git.Repo()
    exp_dir = Path(__file__).parent.resolve()
    train_log_dir_name = "train_log"
    # project_name = "liuce_annotation_with_gyro_direct_fuse_at_2nd_layer"

    def __init__(self, params):
        self.project_name = params.model_name
        self.log_dir = self.exp_dir / 'train_log'

        self.hyper_params = params.hyperparameters
        # if hyper_params:
        #     hcode = ','.join("{}={}".format(k, v) for k, v in hyper_params.items())
        # else:
        #     hcode = 'default'
        # vsha, vnumber = get_latest_commit(self.exp_dir)
        date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = "{}-{}".format(self.project_name, date_time)

        self.log_model_dir = self.log_dir / self.run_id / 'models'
        #self.log_model_dir.mkdir(parents=True, exist_ok=True)

        self.exp_watcher = ExpWatcher(
            project_name=self.project_name,
            experiment_name=self.exp_name,
            run_id=self.run_id,
        )
        self.exp_watcher.upload_hyper_params(self.hyper_params)
        self.exp_log = self.exp_watcher.log_writer("log")
        logger.add(self.exp_log.stream_handler())

        logger.info("{}, {}".format(self.repo.active_branch.name, self.run_id))

    def check_dirty(self):
        if git.Repo().is_dirty():
            logger.warning(
                "\n"
                "-----------------------------------\n"
                "Working tree not clean.\n"
                "You MUST git commit before training.\n"
                "You can use `git commit --amend` if you want to modify unpublished experiment.\n"
                "-----------------------------------"
            )
            return True
        return False


    def write_meta(self, **meta):
        self.exp_watcher.upload_custom_meta_info(meta, merge=True)

    @property
    @functools.lru_cache()
    def exp_name(self) -> str:
        return self.repo.active_branch.name

    @property
    @functools.lru_cache()
    def exp_id(self) -> str:
        return "/".join([self.exp_name, self.run_id])



# vim: ts=4 sw=4 sts=4 expandtab
