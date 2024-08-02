import torch
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
ssl_path = '/mnt/data-home/julian/tiip/dinov2/vits-tiip/model_0082499.rank_0.pth'
ssl_model = torch.load(ssl_path)
ssl_state_dict = ssl_model['model']
prefix = 'teacher.backbone.'
backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14').cuda()
update_ssl_backbone(backbone, ssl_state_dict, prefix=prefix)
import ipdb; ipdb.set_trace()