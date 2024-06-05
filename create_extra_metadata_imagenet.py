from dinov2.data.datasets import ImageNet

root_path ='/mnt/data-home/julian/tiny-imagenet-200'
extra_path = '/mnt/data-home/julian/tiny-imagenet-200-dinov2extra'

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root=root_path, extra=extra_path)
    dataset.dump_extra()