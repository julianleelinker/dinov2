import pathlib
import tqdm


def move_files_to_parent(directory, ext='*.JPEG'):
    # Convert the directory to a Path object
    dir_path = pathlib.Path(directory)
    
    # Iterate through all .jpeg files in the directory and its subdirectories
    for jpeg_file in dir_path.glob(ext):
        # Get the parent directory of the file
        parent_dir = jpeg_file.parent.parent
        
        # Define the new path for the file in the parent directory
        new_path = parent_dir / jpeg_file.name
        
        # Move the file to the parent directory
        jpeg_file.rename(new_path)
        # print(f'Moved {jpeg_file} to {new_path}')
    if not any(dir_path.iterdir()):
        dir_path.rmdir()
        print(f'Removed empty directory: {dir_path}')


def find_subdirectories_and_move_jpeg_files(directory):
    # Convert the directory to a Path object
    dir_path = pathlib.Path(directory)
    
    # Iterate through all subdirectories
    for subdirectory in tqdm.tqdm(dir_path.glob('**/')):
        imgdirectory = subdirectory / 'images'
        if imgdirectory.is_dir():
            print(f'Processing imgdirectory: {imgdirectory}')
            move_files_to_parent(imgdirectory)


root_path ='/mnt/data-home/julian/tiny-imagenet-200'
train_path = f'{root_path}/train'
find_subdirectories_and_move_jpeg_files(train_path)

val_path = f'{root_path}/val'
val_anno_path = f'{val_path}/val_annotations.txt'
with open(val_anno_path, 'r') as f:
    for line in tqdm.tqdm(f):
        line = line.split('\t')
        img_name, class_name = line[0], line[1]
        img_path = f'{val_path}/images/{img_name}'
        class_dir = f'{val_path}/{class_name}'
        pathlib.Path(class_dir).mkdir(parents=True, exist_ok=True)
        new_path = f'{class_dir}/{img_name}'
        pathlib.Path(img_path).rename(new_path)
        print(f'Moved {img_path} to {new_path}')
    if not any(pathlib.Path(f'{val_path}/images').iterdir()):
        pathlib.Path(f'{root_path}/val/images').rmdir()
        print(f'Removed empty directory: {val_path}/images')
