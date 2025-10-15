from torchvision import transforms
import torchvision.transforms.functional as tf
from einops import rearrange
from random import randint
import cv2
import numpy as np
import torch
import os
import random
from monai.transforms.transform import Transform

from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    Resized,
    ToTensord,
    Transpose
)


class ApplyLeveledPatchShuffling(Transform):
    def __init__(self, image_size, patch_size):
        self.image_size = image_size
        self.patch_size = patch_size
        self.vol_length = [self.image_size // self.patch_size] * 3

    def normalize(self, x, newRange=(-3, 3)):  # x is an array. Default range is between zero and one
        xmin, xmax = torch.min(x), torch.max(x)  # get max and min from input array
        norm = (x - xmin) / (xmax - xmin)  # scale between zero and one

        if newRange == (0, 1):
            return (norm)  # wanted range is the same as norm
        elif newRange != (0, 1):
            return norm * (newRange[1] - newRange[0]) + newRange[0]  # scale to a different range.
        # add other conditions here. For example, an error message

    def __call__(self,img, *args, **kwargs):

        perm_gt = []
        for z in range(0, self.image_size // self.patch_size):
            for y in range(0, self.image_size // self.patch_size):
                for x in range(0, self.image_size // self.patch_size):
                    perm_gt.append((z,y,x))

        random.shuffle(perm_gt)
        aug_image = torch.zeros(img["image"].shape)
        counter = 0
        for z in range(0, self.image_size // self.patch_size):
            for y in range(0, self.image_size // self.patch_size):
                for x in range(0, self.image_size // self.patch_size):
                    coords = perm_gt[counter]

                    aug_image[:, z * self.patch_size:z * self.patch_size + self.patch_size,
                    y * self.patch_size:y * self.patch_size + self.patch_size,
                    x * self.patch_size:x * self.patch_size + self.patch_size] = img["image"][:, coords[0] * self.patch_size:coords[0] * self.patch_size + self.patch_size,
                                                                                 coords[1] * self.patch_size:coords[1] * self.patch_size + self.patch_size,
                                                                                 coords[2] * self.patch_size:coords[2] * self.patch_size + self.patch_size]
                    counter+=1
        perm_gt = torch.tensor(perm_gt)

        img["permutation"]= self.normalize(perm_gt)
        img["aug_image"] = aug_image

        return img


class TestTransform(Transform):

    def __init__(self):
        pass
    def __call__(self, image):
        print("image_shape: ",  image["image"].shape)

class PatchCrop(Transform):

    def __init__(self, input_size, patch_size):
        self.input_size = input_size
        self.patch_size = patch_size
        self.patch_needed = int(input_size/ patch_size)

    def get_index(self, a, b):
        # 输入：a为crop1左上角grid的index，b为patch2左上角grid的index
        # 输出：随机挑选出的crop1和crop2对应patch的索引
        (idx_z1, idx_x1, idx_y1), (idx_z2, idx_x2, idx_y2) = a, b

        # 重合部分index范围
        idx_zmin, idx_zmax = max(idx_z1, idx_z2), min((idx_z1 + self.patch_needed), (idx_z2 + self.patch_needed))
        idx_xmin, idx_xmax = max(idx_x1, idx_x2), min((idx_x1 + self.patch_needed), (idx_x2 + self.patch_needed))
        idx_ymin, idx_ymax = max(idx_y1, idx_y2), min((idx_y1 + self.patch_needed), (idx_y2 + self.patch_needed))

        # 找出重合部分在crop1中对应的index list
        overlap_mask_1 = torch.zeros((self.patch_needed, self.patch_needed, self.patch_needed))
        overlap_mask_1[idx_zmin - idx_z1:idx_zmax - idx_z1, idx_xmin - idx_x1:idx_xmax - idx_x1, idx_ymin - idx_y1:idx_ymax - idx_y1] = 1
        overlap_mask_1 = overlap_mask_1.flatten()
        # index1 = torch.nonzero(overlap_mask_1)
        # print(index1)

        overlap_mask_2 = torch.zeros((self.patch_needed, self.patch_needed, self.patch_needed))
        overlap_mask_2[idx_zmin - idx_z2:idx_zmax - idx_z2, idx_xmin - idx_x2:idx_xmax - idx_x2, idx_ymin - idx_y2:idx_ymax - idx_y2] = 1
        overlap_mask_2 = overlap_mask_2.flatten()
        # index2 = torch.nonzero(overlap_mask_2)
        # print(index2)

        return overlap_mask_1.bool(), overlap_mask_2.bool()

    def __call__(self, image):
        if self.patch_size == 32:
            rand_int_end = 1
        elif self.patch_size == 16:
            rand_int_end = 2

        z1 = randint(0,rand_int_end)
        x1 = randint(0,rand_int_end)
        y1 = randint(0,rand_int_end)

        z2 =  randint(0,rand_int_end)
        x2 = randint(0,rand_int_end)
        y2 = randint(0,rand_int_end)


        patch1 = image["image"][:, z1*self.patch_size: (self.patch_needed+z1)*self.patch_size,
                 x1*self.patch_size: (self.patch_needed+x1)*self.patch_size,
                 y1*self.patch_size: (self.patch_needed+y1)*self.patch_size]
        patch2 = image["image"][:, z2*self.patch_size: (self.patch_needed+z2)*self.patch_size,
                 x2*self.patch_size: (self.patch_needed+x2)*self.patch_size,
                 y2*self.patch_size: (self.patch_needed+y2)*self.patch_size]

        image["patch_1"]= patch1
        image["patch_2"]= patch2
        image["overlap_mask_1"], image["overlap_mask_2"] = self.get_index((z1, x1, y1), (z2, x2, y2))
        image["shuffle"] = 0

        return image


class DepthFirst(Transform):
    def __init__(self, depth_first = True):
        self.depth_first = depth_first

    def __call__(self,img, *args, **kwargs):
        C,A,B,X = img["image"].shape
        if self.depth_first and A==B and A!=X and B!=X:
            img["image"] = Transpose((0,3,1,2))(img["image"])
            img["label"] =  Transpose((0,3,1,2))(img["label"])

        return img



def get_loader(config):

    datalist1 = []
    datalist2 = []
    datalist3 = []
    datalist4 = []
    datalist5 = []
    datalist6 = []
    datalist7 = []
    datalist8 = []
    vallist1 = []
    vallist2 = []
    vallist3 = []
    vallist4 = []
    vallist5 = []
    vallist6 = []
    vallist7 = []
    vallist8 = []

    if "luna" in config.dataset:
        temp_datalist = load_decathlon_datalist("3Ddata/LUNA/dataset_LUNA16_sol_scratch.json", False, "training", base_dir="")
        datalist1 = []
        for item in temp_datalist:
            item_dict = {"image": item["image"]}
            datalist1.append(item_dict)

        temp_datalist = load_decathlon_datalist("3Ddata/LUNA/dataset_LUNA16_sol_scratch.json", False, "validation", base_dir="")
        for item in temp_datalist:
            item_dict = {"image": item["image"]}
            vallist1.append(item_dict)

    if "lidc" in config.dataset:
        datalist2 = load_decathlon_datalist("3Ddata/LIDC/dataset_LIDC_sol_scratch.json", False, "training", base_dir="")
        vallist2 = load_decathlon_datalist("3Ddata/LIDC/dataset_LIDC_sol_scratch.json", False, "validation", base_dir="")

    if "tciacolon" in config.dataset:
        temp_datalist = load_decathlon_datalist("3Ddata/TCIAColon/dataset_TCIAColon_sol_scratch.json", False, "training", base_dir="")
        datalist3 = []
        for item in temp_datalist:
            item_dict = {"image": item["image"]}
            datalist3.append(item_dict)

        temp_datalist = load_decathlon_datalist("3Ddata/TCIAColon/dataset_TCIAColon_sol_scratch.json", False, "validation", base_dir="")
        for item in temp_datalist:
            item_dict = {"image": item["image"]}
            vallist3.append(item_dict)

    if "tciacovid" in config.dataset:

        datalist4 = load_decathlon_datalist("3Ddata/TCIACovid/dataset_TCIACovid_sol_scratch.json", False, "training", base_dir="")
        vallist4 = load_decathlon_datalist("3Ddata/TCIACovid/dataset_TCIACovid_sol_scratch.json", False, "validation", base_dir="")

    if "hnscc" in config.dataset:
        temp_datalist = load_decathlon_datalist("3Ddata/HNSCC/dataset_HNSCC_sol_scratch.json", False, "training", base_dir="")
        datalist5 = []
        for item in temp_datalist:
            item_dict = {"image": item["image"]}
            datalist5.append(item_dict)

        temp_datalist = load_decathlon_datalist("3Ddata/HNSCC/dataset_HNSCC_sol_scratch.json", False, "validation", base_dir="")
        for item in temp_datalist:
            item_dict = {"image": item["image"]}
            vallist5.append(item_dict)

    if "oasis" in config.dataset:
        temp_datalist = load_decathlon_datalist("3Ddata/OASIS/dataset_OASIS_sol_scratch.json", False, "training", base_dir="")
        datalist6 = []
        for item in temp_datalist:
            item_dict = {"image": item["image"]}
            datalist6.append(item_dict)

        temp_datalist = load_decathlon_datalist("3Ddata/OASIS/dataset_OASIS_sol_scratch.json", False, "validation", base_dir="")
        for item in temp_datalist:
            item_dict = {"image": item["image"]}
            vallist6.append(item_dict)

    if "amos22" in config.dataset:
        temp_datalist = load_decathlon_datalist("3Ddata/amos22/amos22_sol.json", False, "training", base_dir="")
        datalist7= []
        for item in temp_datalist:
            item_dict = {"image": item["image"]}
            datalist7.append(item_dict)

        temp_datalist = load_decathlon_datalist("3Ddata/amos22/amos22_sol.json", False, "validation", base_dir="")
        for item in temp_datalist:
            item_dict = {"image": item["image"]}
            vallist7.append(item_dict)

    if "btcv" in config.dataset:
        temp_datalist = load_decathlon_datalist("3Ddata/btcv/BTCV_sol.json", False, "training", base_dir="")
        datalist8 = []
        for item in temp_datalist:
            item_dict = {"image": item["image"]}
            datalist8.append(item_dict)

        temp_datalist = load_decathlon_datalist("3Ddata/btcv/BTCV_sol.json", False, "validation", base_dir="")
        for item in temp_datalist:
            item_dict = {"image": item["image"]}
            vallist8.append(item_dict)


    datalist = datalist1 + datalist2 + datalist3 + datalist4 + datalist5 + datalist6 + datalist7 + datalist8
    val_files = vallist1 + vallist2 + vallist3 + vallist4 + vallist5 + vallist6 + vallist7 + vallist8

    print("Total number training data: {}".format(len(datalist)), file=config.log_writter)
    print("Total number validation data: {}".format(len(val_files)), file=config.log_writter)



    if "oasis" in config.dataset:

        train_transforms_consistency = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=(config.space_z, config.space_x, config.space_y),
                    mode=("bilinear"),
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                Resized(keys=["image"], spatial_size=[160, 160, 160], size_mode="all"),
                PatchCrop(patch_size=config.patch_size, input_size=config.image_size),

                ToTensord(keys=["image", "patch_1", "patch_2"]),
            ])


        train_val_transforms_level_ps = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=(config.space_z, config.space_x, config.space_y),
                    mode=("bilinear"),
                ),
                CropForegroundd(keys=["image"], source_key="image"),

                Resized(keys=["image"],spatial_size=[config.image_size, config.image_size, config.image_size],size_mode="all"),
                ApplyLeveledPatchShuffling(image_size=config.image_size, patch_size=config.patch_size),
                ToTensord(keys=["image", "aug_image"]),
            ]
        )
    elif "btcv" or "amos22" in config.dataset:
        train_transforms_consistency = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=(config.space_x, config.space_y,config.space_z),
                    mode=("bilinear"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=config.a_min, a_max=config.a_max, b_min=0, b_max=1, clip=True
                ),
                # CropForegroundd(keys=["image"], source_key="image"),
                Resized(keys=["image"], spatial_size=[160, 160, 160], size_mode="all"),
                PatchCrop(patch_size=config.patch_size, input_size=config.image_size),

                ToTensord(keys=["image", "patch_1", "patch_2", "overlap_mask_1", "overlap_mask_2"]),
            ])

        train_val_transforms_level_ps = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=(config.space_x, config.space_y, config.space_z),
                    mode=("bilinear"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=config.a_min, a_max=config.a_max, b_min=0, b_max=1, clip=True
                ),
                # CropForegroundd(keys=["image"], source_key="image"),

                Resized(keys=["image"], spatial_size=[config.image_size, config.image_size, config.image_size],
                        size_mode="all"),
                ApplyLeveledPatchShuffling(image_size=config.image_size, patch_size=config.patch_size),
                ToTensord(keys=["image", "aug_image"]),
            ]
        )
    else:
        train_transforms_consistency = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=(config.space_z, config.space_x, config.space_y),
                    mode=("bilinear"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=config.a_min, a_max=config.a_max, b_min=0, b_max=1, clip=True
                ),
                #CropForegroundd(keys=["image"], source_key="image"),
                Resized(keys=["image"], spatial_size=[160, 160, 160], size_mode="all"),
                PatchCrop(patch_size=config.patch_size, input_size=config.image_size),

                ToTensord(keys=["image", "patch_1", "patch_2", "overlap_mask_1", "overlap_mask_2"]),
            ])

        train_val_transforms_level_ps = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=(config.space_z, config.space_x, config.space_y),
                    mode=("bilinear"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=config.a_min, a_max=config.a_max, b_min=0, b_max=1, clip=True
                ),
                #CropForegroundd(keys=["image"], source_key="image"),

                Resized(keys=["image"], spatial_size=[config.image_size, config.image_size, config.image_size], size_mode="all"),
                ApplyLeveledPatchShuffling(image_size=config.image_size, patch_size=config.patch_size),
                ToTensord(keys=["image", "aug_image"]),
            ]
        )
    train_ds_consistency = Dataset(data=datalist, transform=train_transforms_consistency)

    train_sampler_consistency = None

    consistency_train_loader = DataLoader(train_ds_consistency, batch_size=config.batch_size, num_workers=config.num_workers,
                              sampler=train_sampler_consistency, shuffle=True, pin_memory=True, drop_last=True)

    train_ds_level_ps = Dataset(data=datalist, transform=train_val_transforms_level_ps)

    train_sampler_level_ps = None

    level_ps_train_loader = DataLoader(train_ds_level_ps, batch_size=config.batch_size, num_workers=config.num_workers,
                              sampler=train_sampler_level_ps, shuffle=True, pin_memory=True, drop_last=True)


    valid_ds = Dataset(data=val_files, transform=train_val_transforms_level_ps)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, num_workers=config.num_workers,shuffle=True,drop_last=True)

    return consistency_train_loader,level_ps_train_loader , valid_loader
















