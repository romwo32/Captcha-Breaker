import os
import shutil
import random

# Set Path to your Image folder
source_dir = r'path-of-all-the-images-that-you-want-to-split'  # Ändere dies zu deinem Bildordner

# Set Path to training,validation and test directories
train_dir = r'path-of-folder-where-you-want-you-taining-data'
validation_dir = r'path-of-folder-where-you-want-you-validation-data'
test_dir = r'path-of-folder-where-you-want-you-test-data'

# Create the directories if not already created
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List of all data in the initial dir
all_images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# shuffle the data in the initial dir
random.shuffle(all_images)

# calculate the amount of pictures for each dataset
num_images = len(all_images)
num_train = int(0.7 * num_images)
num_validation = int(0.15 * num_images)

# split pictures into training, validation and testsets
train_images = all_images[:num_train]
validation_images = all_images[num_train:num_train + num_validation]
test_images = all_images[num_train + num_validation:]

#copy to each dataset to the desired dir
for image in train_images:
    shutil.copy(os.path.join(source_dir, image), train_dir)

for image in validation_images:
    shutil.copy(os.path.join(source_dir, image), validation_dir)

for image in test_images:
    shutil.copy(os.path.join(source_dir, image), test_dir)

print("Your Data is successfully split!")