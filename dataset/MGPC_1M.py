import os
import torch
import random
import math
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

R1 = np.array([
        [1., 0., 0.],
        [0., 0., -1.],
        [0, 1., 0.]
    ])
R2 = np.array([
    [1., 0., 0.],
    [0., -1., 0.],
    [0., 0., -1.]
])


def resize_foreground(image, ratio=0.85):
    assert image.shape[-1] == 4
    alpha_channel = image[..., 3]
    coords = np.where(alpha_channel > 0)
    y1, y2, x1, x2 = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max()
    
    fg = image[y1:y2+1, x1:x2+1]
    
    orig_size = max(fg.shape[0], fg.shape[1])
    final_size = int(orig_size / ratio)
    pad_total = final_size - orig_size
    
    pad_pre = pad_total // 2
    pad_post = pad_total - pad_pre
    vertical_pad = (pad_pre, pad_post)
    horizontal_pad = (pad_pre, pad_post)
    
    new_image = np.pad(
        fg,
        (vertical_pad, horizontal_pad, (0, 0)),
        mode="constant",
        constant_values=0
    )
    
    return new_image



class MGPC_1M(data.Dataset):
    def __init__(self, data_path, subset):

        self.data_root = data_path
        self.subset = subset
        self.data_spilt_path = os.path.join(self.data_root, self.subset)

        self.file_list = []

        print(f'[DATASET] Dataset Path: {self.data_spilt_path}')
        for category_folder in os.listdir(self.data_spilt_path):
            category_path = os.path.join(self.data_spilt_path, category_folder)
            for obj in os.listdir(category_path):
                name, ext = os.path.splitext(obj)
                if ext.lower() == ".png":
                    input_rgb_path = os.path.join(category_path, obj)
                    gt_pc_path = os.path.join(category_path, category_folder + "_gt.npy")
                    input_data_path = os.path.join(category_path, name + ".npz")
                    self.file_list.append({
                        "input_rgb_path": input_rgb_path,
                        "input_data_path": input_data_path,
                        "gt_pc_path": gt_pc_path
                    })
        self.dataset_length = len(self.file_list)
        print(f'[DATASET] Dataset Pairs Number: {self.dataset_length}')

        self.rgb_normalize = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def rotate_rgb(self, img, angle):
        img = transforms.functional.rotate(img, -angle, fill=(71, 71, 71), expand=False)
        return img


    def pc_normalization(self, input_data, gt_pc, angle):
        input_points = input_data['points']
        centroid = input_data['centroid']
        scale = input_data['scale']
        rotation = input_data['rotation']
        position = input_data['position']
        # Input-Centric Normalization: align gt to input pc
        gt_points = np.dot(gt_pc, R1.T)
        gt_points = gt_points - position
        gt_points = np.dot(gt_points, rotation)
        gt_points = gt_points - centroid
        gt_points = np.dot(gt_points, R2.T)
        gt_points = gt_points * scale

        # align axis with real camera
        input_points = np.dot(input_points, R2.T)
        # random rotation
        if angle != 0:
            angle_rad = math.radians(angle)
            rot_mat = np.array([
                [math.cos(angle_rad), -math.sin(angle_rad), 0],
                [math.sin(angle_rad),  math.cos(angle_rad), 0],
                [0,                    0,                   1]
            ])
            input_points = np.dot(input_points, rot_mat.T)
            gt_points = np.dot(gt_points, rot_mat.T)

        input_points = torch.from_numpy(input_points).float()
        gt_points = torch.from_numpy(gt_points).float()
        return input_points, gt_points

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        input_rgb_path = sample["input_rgb_path"]
        input_data_path = sample["input_data_path"]
        gt_pc_path = sample["gt_pc_path"]

        gt_pc = np.load(gt_pc_path, allow_pickle=True)
        input_data = np.load(input_data_path, allow_pickle=True)

        # Normalization and augmentation
        angle = random.uniform(0, 90) if self.subset.lower() == "train" else 0
        normalized_input_pc, normalized_gt_pc = self.pc_normalization(input_data, gt_pc, angle)

        # rgb
        input_rgb = Image.open(input_rgb_path).convert('RGB')
        if angle != 0:
            input_rgb = self.rotate_rgb(input_rgb, angle)
        normalized_rgb = self.rgb_normalize(input_rgb)
        
        # text
        name = os.path.basename(gt_pc_path).replace('_gt.npy', '').lower()
        words = name.split('_')
        if words[-1].isdigit():
            words.pop()
        obj_name = ' '.join(words).rstrip(' ')
        prompt = f"This is a/an \"{obj_name}\""

        return normalized_input_pc, normalized_gt_pc, normalized_rgb, prompt, input_data_path

    def __len__(self):
        return len(self.file_list)