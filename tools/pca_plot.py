import torch
import os
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca_plot(backbone, use_vit, images_path, feat_dim, title='pca_all_images', feature_key='patch_tokens', image_size = 224, patch_size = 14, yolo_od_model=False):
    background_threshold = float('-inf') 
    smaller = True

    transform0 = transforms.Compose([           
                                    transforms.Resize(256),                    
                                    transforms.CenterCrop(224),               
                                    transforms.ToTensor(),                    
                                    transforms.Normalize(                      
                                    mean=[0.485, 0.456, 0.406],                
                                    std=[0.229, 0.224, 0.225]              
                                    )])


    transform1 = transforms.Compose([           
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size), #should be multiple of model patch_size                 
                                    transforms.ToTensor(),                    
                                    transforms.Normalize(mean=0.5, std=0.2)
                                    ])

    patch_h  = image_size//patch_size
    patch_w  = image_size//patch_size


    total_features  = []
    with torch.no_grad():
        for img_path in os.listdir(images_path):
            img_path = os.path.join(images_path, img_path)
            img = Image.open(img_path).convert('RGB')
            img_t = transform1(img)
            if use_vit:
                features_dict = backbone.forward_features(img_t.unsqueeze(0).cuda())
            else:
                features_dict = backbone(img_t.unsqueeze(0).cuda())
            features = features_dict[feature_key]
            total_features.append(features)

    total_features = torch.cat(total_features, dim=0).cpu().detach()
    print('*'*50)
    print(f'total features shape: {total_features.shape}')

    if yolo_od_model:
        total_features = torch.reshape(total_features, (total_features.shape[0], total_features.shape[1], -1))
        total_features = torch.permute(total_features, (0, 2, 1))

    total_features = total_features.reshape(4 * patch_h * patch_w, feat_dim) 

    pca = PCA(n_components=3, whiten=False)
    pca.fit(total_features)
    pca_features = pca.transform(total_features)
    print('*'*50)
    print(f'PCA features shape: {pca_features.shape}')


    # visualize PCA components for finding a proper threshold
    # 3 histograms for 3 components
    fig, axis = plt.subplots(4, 4, figsize=(12, 12))
    row = 0
    for col in range(3):
        axis[row][col].hist(pca_features[:, col])


    # min_max scale
    # pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
    #                      (pca_features[:, 0].max() - pca_features[:, 0].min())
    #pca_features = sklearn.processing.minmax_scale(pca_features)

    row = 1
    for col in range(4):
        axis[row][col].imshow(pca_features[col*patch_h*patch_w : (col+1)*patch_h*patch_w, 0].reshape(patch_h, patch_w))


    #  cell 10
    # segment/seperate the backgound and foreground using the first component
    if smaller:
        pca_features_bg = pca_features[:, 0] < background_threshold # from first histogram
    else:
        pca_features_bg = pca_features[:, 0] > background_threshold # from first histogram

    pca_features_fg = ~pca_features_bg

    # fig, axis = plt.subplots(1, 4, figsize=(12, 3))
    # fig.suptitle('background filtered by PCA first component', fontsize=12)
    # for i in range(4):
    #     axis[i].imshow(pca_features_bg[i*patch_h*patch_w : (i+1)*patch_h*patch_w].reshape(patch_h, patch_w))
    # plt.show()


    # cell 11

    # 2nd PCA for only foreground patches
    pca.fit(total_features[pca_features_fg]) 
    pca_features_left = pca.transform(total_features[pca_features_fg])

    for i in range(3):
        # min_max scaling
        pca_features_left[:, i] = (pca_features_left[:, i] - pca_features_left[:, i].min()) / (pca_features_left[:, i].max() - pca_features_left[:, i].min())

    pca_features_rgb = pca_features.copy()
    # for black background
    pca_features_rgb[pca_features_bg] = 0
    # new scaled foreground features
    pca_features_rgb[pca_features_fg] = pca_features_left

    # reshaping to numpy image format
    pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)

    # title = 'pca_rgb'
    row = 2
    for col, img_path in enumerate(os.listdir(images_path)):
        axis[row][col].imshow(pca_features_rgb[col])


    row = 3
    for col, img_path in enumerate(os.listdir(images_path)):
        img_path = os.path.join(images_path, img_path)
        img = Image.open(img_path).convert('RGB').resize((image_size, image_size))
        axis[row][col].imshow(img)
    plt.savefig(f'{title}.png')
