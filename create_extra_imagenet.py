from dinov2.data.datasets import ImageNet

root_path = '/mnt/data-home/julian/OpenDataLab___ImageNet-1K/raw/ImageNet-1K'
extra_path = '/mnt/data-home/julian/ImageNet-1K-dino-extra'

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root=root_path, extra=extra_path)
    dataset.dump_extra()