import os
import torch
import cv2
import glob

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from abc import abstractmethod

from transform import transforms_lib
from loss.loss import upsample2d_flow_as
from utils.flow_utils import homo_to_flow


class BaseDataset(Dataset):
    def __init__(self, frame_pair_path, input_transform, spatial_transform, with_gt_flow=None, evaluate_with_whole_img=None):
        self.frame_pair_path = os.path.join(frame_pair_path)

        self.input_transform = input_transform
        self.spatial_transform = spatial_transform

        self.samples = self.collect_samples()
        self.gyro_homos = np.load(os.path.join(self.frame_pair_path, "gyro_homo_h33_source_no_nan.npy"))

        self.with_gt_flow = with_gt_flow
        self.evaluate_with_whole_img = evaluate_with_whole_img

    @abstractmethod
    def collect_samples(self):
        pass

    def __len__(self):
        return len(self.samples[0])

    def __getitem__(self, idx):
        frame_path_1 = self.samples[0][idx]
        frame_path_2 = self.samples[1][idx]
        gyro_homo = self.gyro_homos[idx]

        if self.with_gt_flow:
            gt_flow = np.load(self.samples[2][idx], allow_pickle=True)
            gt_flow = torch.from_numpy(gt_flow).permute(2, 0, 1)
            # read corrsponding label
            try:
                # 验证集暂时没有打label
                label = self.samples[3][idx]
                rain_label = self.samples[4][idx]
            except:
                label = "Nil"
                rain_label = "Nil"
        try:
            imgs = [cv2.imread(i).astype(np.float32) for i in [frame_path_1, frame_path_2]]
        except:
            print(frame_path_1 + " " + frame_path_2)
            raise Exception

        # gyro_homo is the homography from img1 to img2
        gyro_filed = transforms_lib.homo_to_flow(np.expand_dims(gyro_homo, 0))[0]

        dummy_data = torch.zeros([1, 2, 512, 640])

        # if self.weather_transform:
        #     if random.random() < 0.5:
        #         # cv2.imwrite("without_weather.png", imgs[0])
        #         imgs = [self.weather_transform(image=i) for i in imgs]
        #         # cv2.imwrite("with_weather.png", imgs[0])

        if self.evaluate_with_whole_img:
            imgs = [cv2.resize(i, (640, 512)) for i in imgs]
            gyro_filed = upsample2d_flow_as(gyro_filed, dummy_data, if_rate=True).squeeze()

        if self.spatial_transform is not None:
            imgs.append(gyro_filed.squeeze().permute(1, 2, 0))
            data = self.spatial_transform(imgs)
            imgs, gyro_filed = data[:2], data[-1]
            gyro_filed = gyro_filed.permute(2, 0, 1)

        if self.input_transform:
            imgs_it = [self.input_transform(i) for i in imgs]

        ret = {"img{}".format(i + 1): v for i, v in enumerate(imgs_it)}

        ret["gyro_field"] = gyro_filed

        if self.with_gt_flow:
            ret["gt_flow"] = gt_flow
            ret["label"] = label
            ret["rain_label"] = rain_label
        return ret


class FrameDataset(BaseDataset):
    def __init__(self, frame_pair_path, input_transform, spatial_transform, with_gt_flow, evaluate_with_whole_img, weather_transform=None):
        super(FrameDataset, self).__init__(frame_pair_path=frame_pair_path,
                                           input_transform=input_transform,
                                           spatial_transform=spatial_transform,
                                           with_gt_flow=with_gt_flow,
                                           evaluate_with_whole_img=evaluate_with_whole_img)
        self.weather_transform = weather_transform

    def collect_samples(self):
        return np.load(os.path.join(self.frame_pair_path, "frames.npy"), allow_pickle=True)


class FrameDatasetV2(Dataset):
    def __init__(self, input_transform, spatial_transform):
        self.input_transform = input_transform
        self.spatial_transform = spatial_transform

        self.samples = self.collect_samples()

    def collect_samples(self):
        files = glob.glob("dataset/samples/sample*")
        return files

    def resize_flow(self, inputs, target_as, isRate=False):
        h, w, _ = target_as.shape
        h_, w_, _ = inputs.shape
        res = cv2.resize(inputs, (w, h), interpolation=cv2.INTER_LINEAR)
        if isRate:
            u_scale = (w / w_)
            v_scale = (h / h_)
            res[:, :, 0] *= u_scale
            res[:, :, 1] *= v_scale
        return res

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file = self.samples[idx]
        frame_path_1 = os.path.join(file, "img1.png")
        frame_path_2 = os.path.join(file, "img2.png")

        gyro_homo_path = os.path.join(file, "gyro_homo.npy")
        gyro_homo = np.load(gyro_homo_path)

        try:
            imgs = [cv2.imread(i).astype(np.float32) for i in [frame_path_1, frame_path_2]]
        except Exception as e:
            print(frame_path_1 + " " + frame_path_2)
            raise e

        # gyro_homo is the homography from img1 to img2
        gyro_filed = homo_to_flow(np.expand_dims(gyro_homo, 0), H=600, W=800).squeeze()

        imgs.append(gyro_filed)
        data = self.spatial_transform(imgs)
        imgs, gyro_filed = data[:2], data[-1]
        gyro_filed = gyro_filed.transpose(2, 0, 1)

        imgs_it = [self.input_transform(i) for i in imgs]

        ret = {"img{}".format(i + 1): v for i, v in enumerate(imgs_it)}
        ret["gyro_field"] = torch.from_numpy(gyro_filed)
        return ret


class TestDataset(Dataset):
    def __init__(self, benchmark_path, input_transform):
        self.input_transform = input_transform

        self.samples = np.load(benchmark_path, allow_pickle=True)

        self.masks_pth = "/data/fusion_gyro_optical_flow/deep_gyro_of_fusion/work/20220602.GyroHomo.Naive/benchmark/mask"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # dummy_data = torch.zeros([1, 2, 512, 640])

        imgs = [self.samples[idx]["img1"], self.samples[idx]["img2"]]

        gyro_homo = self.samples[idx]["homo"]

        gt_flow = self.samples[idx]["gt_flow"]
        gt_flow = torch.from_numpy(gt_flow).permute(2, 0, 1)

        mask_path = os.path.join(self.masks_pth, "mask_out_{}.png".format(idx))
        try:
            mask = cv2.imread(mask_path)[:, :, :1] / 255
        except Exception as e:
            print(mask_path)
            raise e
        mask = 1 - mask
        mask = torch.from_numpy(mask).permute(2, 0, 1)

        split = self.samples[idx]["split"]

        gyro_filed = transforms_lib.homo_to_flow(np.expand_dims(gyro_homo, 0), H=600, W=800)[0]

        gyro_filed = gyro_filed.squeeze()

        if self.input_transform:
            imgs_it = [self.input_transform(i) for i in imgs]

        ret = {"img{}".format(i + 1): v for i, v in enumerate(imgs_it)}

        ret["gyro_field"] = gyro_filed
        ret["gt_flow"] = gt_flow
        ret["label"] = split
        ret["mask"] = mask
        ret["rain_label"] = split
        return ret


def fetch_dataloader(types, status_manager):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        status_manager: (class) status_manager

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    input_transform = transforms_lib.fetch_input_transform()
    spatial_transform = transforms_lib.fetch_spatial_transform(status_manager.params)
    weather_transform = transforms_lib.weather_transform()

    for split in ['train', 'valid', 'test', 'train_nori']:
        if split in types:
            frame_path = os.path.join(status_manager.params.data_dir, "{}".format(split))

            benchmark_path_gof_clean = "/data/fusion_gyro_optical_flow/deep_gyro_of_fusion/work/SGF_ablation_study/dataset/GHOF_Clean_20230705.npy"
            benchmark_path_gof_final = "/data/fusion_gyro_optical_flow/deep_gyro_of_fusion/work/SGF_ablation_study/dataset/GHOF_Final_20230705.npy"

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                # ds = ConcatDataset([
                #     FrameDataset(frame_path,
                #                  input_transform=input_transform,
                #                  spatial_transform=spatial_transform,
                #                  with_gt_flow=None,
                #                  evaluate_with_whole_img=None,
                #                  weather_transform=None),
                #     FrameDatasetV2(input_transform=input_transform, spatial_transform=spatial_transform)
                # ])
                ds = FrameDatasetV2(input_transform=input_transform, spatial_transform=spatial_transform)
                dl = DataLoader(ds,
                                batch_size=status_manager.params.hyperparameters.batch_size,
                                shuffle=True,
                                num_workers=status_manager.params.num_workers,
                                pin_memory=status_manager.params.cuda)

            elif split == "valid":
                dl = DataLoader(FrameDataset(frame_path,
                                             input_transform=input_transform,
                                             spatial_transform=None,
                                             with_gt_flow=True,
                                             evaluate_with_whole_img=True),
                                batch_size=1,
                                shuffle=False,
                                num_workers=status_manager.params.num_workers,
                                pin_memory=status_manager.params.cuda)
            elif split == "test":
                input_transform = transforms_lib.fetch_input_transform(if_normalize=False)
                ds = ConcatDataset([
                    TestDataset(benchmark_path_gof_clean, input_transform=input_transform),
                    TestDataset(benchmark_path_gof_final, input_transform=input_transform)
                ])
                dl = [
                    DataLoader(s,
                               batch_size=1,
                               shuffle=False,
                               num_workers=status_manager.params.num_workers,
                               pin_memory=status_manager.params.cuda) for s in ds.datasets
                ]
            else:
                raise Exception()

            dataloaders[split] = dl
    return dataloaders
