import torch
from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup
from dinov2.train import get_args_parser
from pca_plot import pca_plot


def update_ssl_backbone(dst_model, src_state_dict, prefix):
    print(f"updating backbone")
    updated_count = 0
    unupdated_count = 0
    for key in dst_model.state_dict():
        src_model_key = prefix + key
        if src_model_key in src_state_dict:
            print(f'{src_model_key} in src model')
            updated_count += 1
            dst_model.state_dict()[key].copy_(src_state_dict[src_model_key])
        else:
            print(f'{src_model_key} not in src model')
            unupdated_count += 1
    print(f'{updated_count=} {unupdated_count=}')


def main(args):
    cfg = setup(args)
    ssl_path = '/mnt/data-home/julian/tiip/dinov2/vits14-tiip/model_final.rank_0.pth'
    ssl_model = torch.load(ssl_path)
    ssl_state_dict = ssl_model['model']
    teacher_backbone, teacher_embed_dim = build_model_from_cfg(cfg, only_teacher=True)
    update_ssl_backbone(teacher_backbone, ssl_state_dict, prefix='teacher.backbone.')
    teacher_backbone.cuda()

    # this threshold depends on model size, and not sure the criterion is < or >
    # currently tested < for b, g, > for l, s
    # images_path = '/home/julian/work/Dino_V2/harryported_giffin_images'
    images_path = '/mnt/data-home/julian/ssl/pca-examples/tiip-s1'
    # images_path = '/mnt/data-home/julian/ssl/pca-examples/fashion'
    # images_path = '/mnt/data-home/julian/ssl/pca-examples/imagenet'

    pca_plot(teacher_backbone, use_vit=True, title='vits_ssl_tiip',images_path='/mnt/data-home/julian/ssl/pca-examples/tiip-s1', feat_dim=384, feature_key='x_norm_patchtokens', image_size=518, patch_size=14)
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)