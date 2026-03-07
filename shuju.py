import os
import shutil
import random
import cv2

### Define a function to split images into train and val folders
def split_images(input_folder, output_folder, split_ratio=0.9):
    train_folder = os.path.join(output_folder, 'train')
    validate_folder = os.path.join(output_folder, 'val')
    num = 0

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validate_folder, exist_ok=True)

    ### Get all subfolders (each representing a pest class)
    subfolders = [f.name for f in os.scandir(input_folder) if f.is_dir()]

    ### Iterate over each pest category
    for subfolder in subfolders:
        subfolder_path = os.path.join(input_folder, subfolder)
        train_subfolder_path = os.path.join(train_folder, subfolder)
        validate_subfolder_path = os.path.join(validate_folder, subfolder)

        os.makedirs(train_subfolder_path, exist_ok=True)
        os.makedirs(validate_subfolder_path, exist_ok=True)

        images = [f.name for f in os.scandir(subfolder_path)
                  if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        num_images = len(images)
        num_validate = int(num_images * (1 - split_ratio))

        ### Randomly select images for validation
        validate_images = random.sample(images, num_validate)

        ### Copy or save images into the appropriate folders
        for image in images:
            source_path = os.path.join(subfolder_path, image)
            img = cv2.imread(source_path)
            name = str(num) + ".png"

            if img is not None:
                if image in validate_images:
                    destination_path = os.path.join(validate_subfolder_path, name)
                else:
                    destination_path = os.path.join(train_subfolder_path, name)
                cv2.imwrite(destination_path, img)
            else:
                print("Invalid image or file not found.")
            num += 1

### Run the splitting function
input_folder = "D:/毕业设计/pests"
output_folder = "D:/毕业设计/datasets"
split_images(input_folder, output_folder)
