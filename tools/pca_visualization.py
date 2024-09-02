# %%
import torch
import os
import copy
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import yaml_model_load, parse_model, DetectionModel
from pca_plot import pca_plot


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
title = 'yolov8m_ssl_imagenet'
use_vit = False
yolo_od_model = False
show_all_diagrams = True

feature_key = 'x_norm_patchtokens'
if use_vit:
    # background_threshold = -50
    background_threshold = float('-inf') 
    smaller = True
    whiten = False
    # settings for ViT
    feat_dim_dict = {'s': 384, 'b': 768, 'l': 1024, 'g': 1536}
    # model_size in ['s', 'b', 'l', 'g']
    model_size = 's' 
    feat_dim = feat_dim_dict[model_size]
    backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14').cuda()
    image_size = 518
    patch_size = backbone.patch_size # patchsize=14

else:
    background_threshold = float('-inf') 
    smaller = True
    whiten = False
    # settings for YOLO
    yolo_path = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/models/v8/yolov8m-ssl.yaml'
    # yolo_path = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/models/v8/yolov8m-neck.yaml'
    # yolo_path = '/home/julian/work/dinov2/ultralytics/ultralytics/cfg/models/v8/yolov8x-neck.yaml'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/imangenet-v0.1-0712/model_final.pth'
    # ssl_path = '/home/julian/yolov8n-scratch.pt'
    # ssl_path = '/home/julian/work/dinov2/yolov8n.pt'
    # ssl_path = '/home/julian/work/dinov2/yolov8m.pt'
    # ssl_path = '/home/julian/work/dinov2/yolov8x.pt'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/tiip-yolov8m-v0.1.1-0720/model_final.pth'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/tiip-yolov8m-v0.1.1-0722/model_final.pth'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/tiip-yolov8m-v0.1.2-0722-2/model_0008699.pth'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/tiip-yolov8m-v0.1.2-0722-2/model_final.pth'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/resize_640/model_0005549.pth'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/resize_640_global_512_local_224_0726/model_0003899.pth'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/resize_640_global_224_local_98_0807/model_0009999.pth'
    # ssl_path = '/mnt/data-home/julian/tiip/dinov2/resize_640_global_512_local_224_0726/model_final.pth'
    # ssl_path = '/home/julian/work/dinov2/scratch-yolov8m-07222-last.pt'
    ssl_path = '/mnt/data-home/julian/tiip/dinov2/yolo-dis-imagenet/model_0019999.rank_0.pth'
    yolo_yaml = yaml_model_load(yolo_path) 
    feat_dim, patch_size, idx = 576, 32, 2
    # feat_dim, patch_size, idx = 384, 16, 1
    # feat_dim, patch_size, idx = 192, 8, 0

    # for yolov8x
    # feat_dim, patch_size, idx = 640, 32, 2
    # feat_dim, patch_size, idx = 640, 16, 1
    # feat_dim, patch_size, idx = 320, 8, 0
    image_size = 640

    def update_ssl_backbone(yolo_model, ssl_state_dict, prefix):
        print(f"updating backbone {ssl_path}")
        updated_count = 0
        unupdated_count = 0
        for key in yolo_model.state_dict():
            ssl_key = prefix + key
            if ssl_key in ssl_state_dict:
                # print(f'{ssl_key} in ssl model')
                updated_count += 1
                yolo_model.state_dict()[key].copy_(ssl_state_dict[ssl_key])
            else:
                print(f'{ssl_key} not in ssl model')
                unupdated_count += 1
        print(f'{updated_count=} {unupdated_count=}')

    
    ch = 3
    # model, save = parse_model(copy.deepcopy(yaml), ch=ch, verbose=True)  # model, savelist
    yolo_model, _ = parse_model(copy.deepcopy(yolo_yaml), ch=ch, verbose=True)
    ssl_model = torch.load(ssl_path)

    if yolo_od_model:
        yolo_model = DetectionModel(cfg=yolo_path).cuda()
        # ssl_model = YOLO("yolov8m.yaml").load(ssl_path)
        ssl_model = YOLO("yolov8x.yaml").load(ssl_path)
        ssl_state_dict = ssl_model.model.state_dict()
        prefix = 'model.'
        update_ssl_backbone(yolo_model.model, ssl_state_dict, prefix=prefix)
        feature_key = idx
        print(feature_key)
    else:
        ssl_state_dict = ssl_model['model']
        prefix = 'teacher.backbone.'
        update_ssl_backbone(yolo_model, ssl_state_dict, prefix=prefix)
    backbone = yolo_model
    backbone.cuda()

pca_plot(backbone, use_vit=use_vit, title=title, images_path='/mnt/data-home/julian/ssl/pca-examples/tiip-s1', feat_dim=feat_dim, feature_key=feature_key, image_size=image_size, patch_size=patch_size)
