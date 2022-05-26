import json
import os
import glob
import cv2
import scipy.misc as m
import numpy as np
import copy

import torch
import torch.nn as nn

import model_selector
import datasets


def process_depth(dep, a_min=0, a_max=10):
    dep = np.clip(dep, a_min=a_min, a_max=a_max)
    dep = (copy.deepcopy(dep) - a_min) / (a_max - a_min)
    dep_vis = dep * 255

    return dep_vis.astype('uint8')


with open('configs.json') as config_params:
    configs = json.load(config_params)

with open("sample.json") as json_params:
    params = json.load(json_params)
        
root_dir = configs['cityscapes']['path']
image_path= os.path.join(root_dir, "leftImg8bit/test/berlin/")
depth_path = os.path.join(root_dir, "disparity/test/berlin/")



model = model_selector.get_model(params)
print(model.keys())
tasks = params['tasks']

stat_dict = torch.load("/data/lyma/MTL/optimizer=Adam|batch_size=8|lr=0.0005|dataset=cityscapes|normalization_type=none|algorithm=mgda|use_approximation=True|parallel=False_3_model.pkl")
print(stat_dict.keys())

# for k in 
model['rep'].load_state_dict(stat_dict['model_rep'])
model['rep'].eval()
model['S'].load_state_dict(stat_dict['model_S'])
model['S'].eval()
model['I'].load_state_dict(stat_dict['model_I'])
model['I'].eval()
model['D'].load_state_dict(stat_dict['model_D'])
model['D'].eval()

mean = np.array([123.675, 116.28, 103.53])
img_size=(configs['cityscapes']['img_rows'], configs['cityscapes']['img_cols'])
train_loader, train_dst, val_loader, val_dst, test_loader, test_dst = datasets.get_dataset(params, configs)
result = dict()
with torch.no_grad():
    # for img_path in glob.glob(image_path+"*.png"):
    #     print(img_path)
    #     img = m.imread(img_path)
    #     img = img[:, :, ::-1]
    #     img = img.astype(np.float64)
    #     img -= mean
    #     img = m.imresize(img, (img_size[0], img_size[1]))
    #     # Resize scales images from 0 to 255, thus we need
    #     # to divide by 255.0
    #     img = img.astype(float) / 255.0
    #     # NHWC -> NCWH
    #     img = img.transpose(2, 0, 1)
    #     img = torch.from_numpy(img).float().unsqueeze(0).cuda()
    #     print(img.shape)
    for images, lbl, ins_gt, depth in val_loader:
        images = images.cuda()
        rep, mask = model['rep'](images, None)
        for t in tasks:
            result[t], _ = model[t](rep, None)
            print(t, result[t].shape)
        pred_depth = result['D']
        pred_depth = nn.functional.interpolate(
                pred_depth,
                (img_size[0], img_size[1]),
                mode='bilinear', align_corners=True).squeeze().cpu().numpy()
        print(pred_depth.shape, pred_depth[0,:,:].shape)
        cv2.imwrite("depth.jpg", process_depth(pred_depth[0,:,:], np.min(pred_depth[0,:,:]), np.max(pred_depth[0,:,:])) )
        # cv2.imwrite("depth.jpg", process_depth(pred_depth, np.min(pred_depth), np.max(pred_depth)) )
        exit()
