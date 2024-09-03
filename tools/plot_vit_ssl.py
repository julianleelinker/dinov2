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
    title = 'test'
    model_type = 'full'

    # feat_dim, ssl_path = 384, '/mnt/data-home/julian/tiip/dinov2/vits14-tiip/model_final.rank_0.pth'
    feat_dim, ssl_path = 1024, '/mnt/data-home/julian/tiip/dinov2/hungyu/model_0059993.rank_0.pth'
    # feat_dim, ssl_path = 1024, '/mnt/data-home/julian/tiip/dinov2/hungyu/teacher_checkpoint.pth'

    ssl_model = torch.load(ssl_path)

    if model_type == 'full':
        ssl_state_dict = ssl_model['model']
        prefix = 'teacher.backbone.'
    elif model_type == 'teacher':
        ssl_state_dict = ssl_model['teacher']
        prefix = 'backbone.'

    teacher_backbone, teacher_embed_dim = build_model_from_cfg(cfg, only_teacher=True)
    update_ssl_backbone(teacher_backbone, ssl_state_dict, prefix=prefix)
    teacher_backbone.cuda()

    # this threshold depends on model size, and not sure the criterion is < or >
    # currently tested < for b, g, > for l, s
    # image_path = '/home/julian/work/Dino_V2/harryported_giffin_images'
    # image_path = '/mnt/data-home/julian/ssl/pca-examples/fashion'
    # image_path = '/mnt/data-home/julian/ssl/pca-examples/imagenet'

    image_path = '/mnt/data-home/julian/ssl/pca-examples/tiip-s1'

    pca_plot(teacher_backbone, use_vit=True, title=title,images_path=image_path, feat_dim=feat_dim, feature_key='x_norm_patchtokens', image_size=518, patch_size=14)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)