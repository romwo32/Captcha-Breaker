# TensorFlow CRNN model for reading Captchas
Training a Custom CRNN for Captcha Image Text Extraction with TensorFlow and CTC Loss Function: A Step-by-Step Guide.

Captchas (Completely Automated Public Turing Test to Tell Computers and Humans Apart) are used to protect websites from bots and automated scripts by presenting a challenge that is easy for humans to solve but difficult for computers. One common type of captcha is a simple image containing a sequence of letters or numbers the user must enter to proceed.

This model is a solution using TensorFlow and the Connectionist Temporal Classification (CTC) loss function. And not to write everything from scratch, I'll use "indert github repo" as a basis.

Prerequisites:
You will need to have the following software installed:

Python 3.7+
TensorFlow 2.x (pip install tensorflow)
OpenCV (pip install opencv-python)
NumPy (pip install numpy)
Matplotlib (pip install matplotlib)
Google Colab (if running on Colab, no installation needed)
Dataset of your choice

Download the Captcha dataset:
After installing the required packages, we can download the dataset we'll use to train our model. Dataset you can download from this repository. The dataset contains 5000 captcha files as jpg images. The label for each sample is a string, the file's name excluding the file extention.

## Uploading & Extracting the Dataset
To simplify things, the Code incluedes a script that can upload files into Google Colab from your local System:

uploaded = files.upload()
# Extrahiere die hochgeladene ZIP-Datei
for file_name in uploaded.keys():
    if file_name.endswith(".zip"):
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall("/content")
        print(f"{file_name} wurde extrahiert!")

The next step is to extract the Uploaded Zip file to usable path. The following script extracts the images from a Zip Folder with the following structure:
ZipFolder/
│── Folder/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...

for file_name in uploaded.keys():
    if file_name.endswith(".zip"):
        extraction_path = f"/content/{file_name.replace('.zip', '')}_extracted"

        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            # Print first few file names to understand structure
            print("First 10 files in ZIP:", zip_ref.namelist()[:10])

            # Extract all files, preserving directory structure
            zip_ref.extractall(extraction_path)

        # Find all JPG files recursively
        image_paths = list(Path(extraction_path).rglob("*.jpg"))

        print(f"Extracted to: {extraction_path}")
        print("Number of images found:", len(image_paths))
        print("First few image paths:", image_paths[:5])

# If images found, update your image directory
if image_paths:
    image_dir = Path(extraction_path)


# Preprocess the dataset:
Data Preprocessing for CRNN
To train our CRNN (Convolutional Recurrent Neural Network) on CAPTCHA images, we first need to preprocess the dataset. This involves defining constants, loading images and labels, encoding characters, and creating TensorFlow datasets.

Define Constants

Set image height, width, and batch size.
Load Image Files & Labels

Specify the directory containing CAPTCHA images.
Extract filenames as labels (text from CAPTCHA).
Determine Maximum Label Length

Find the longest CAPTCHA text for standardization.
Create Character Mappings

Extract unique characters from all labels.
Map characters to integers and vice versa.
Preprocess Images

Convert to grayscale, resize, and enhance contrast.
Encode Labels

Convert label text into integer sequences.
Create TensorFlow Dataset

Pair preprocessed images with encoded labels.
Shuffle dataset to improve training.
Split into Training & Validation Sets

Use 80% for training and 20% for validation.
Batch, shuffle, and optimize data loading.
This ensures clean and efficient data preparation for training the CRNN model. 



