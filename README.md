# Captcha-Breaker
Exploring various Deep Learning approaches to solve CAPTCHA-related challenges. This repository includes model implementations and evaluations for bypassing Text Recognition CAPTCHAs using neural networks like CNN, CRNN & LigthGBM. 🚀🔓
# What are Captchas? 
A CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) is a type of challenge-response test used to determine whether the user is a human or a bot. It is commonly used on websites and online services to prevent automated programs (bots) from performing actions that could be harmful or undesirable, such as spamming, brute-force attacks, or data scraping.

CAPTCHAs typically present puzzles that are easy for humans to solve but difficult for machines. Some common types of CAPTCHAs include:

Image Recognition: Users are asked to select images that contain certain objects, such as traffic lights, bicycles, or animals.
Text Recognition: Users are shown distorted text or numbers and asked to type them into a box.
Audio CAPTCHA: A series of spoken letters or numbers that the user must enter.
ReCAPTCHA: A more advanced system, often used by Google, that not only tests human interaction but also helps improve machine learning by analyzing user behaviors.
CAPTCHAs help safeguard websites, prevent malicious actions, and protect online services from abuse by ensuring that only real humans can perform certain activities.

# Why It's Important to Understand How Models Solve Our CAPTCHAs

Understanding how models solve CAPTCHAs is essential for several reasons, particularly when protecting valuable assets:

Security Validation of Text-Based CAPTCHAs: By analyzing how models approach CAPTCHA challenges, we can assess their robustness and identify potential vulnerabilities. This ensures that CAPTCHAs are strong enough to prevent automated systems from bypassing them, especially when it comes to securing assets like Bitcoin.

Identifying Deep Learning Capabilities: Understanding how advanced machine learning techniques, including deep learning, can break through text-based CAPTCHAs is crucial. This insight helps us stay ahead of evolving AI technologies and strengthens the resistance of CAPTCHAs against sophisticated attacks.

Exploring Application Potential: Researching how models solve CAPTCHAs can serve as a foundation for developing more robust and secure CAPTCHA solutions. By uncovering weaknesses, we can improve the security and reliability of systems used to protect sensitive digital assets.

By ensuring that CAPTCHAs remain effective against evolving AI models, we help protect critical systems from bots and unauthorized access, ultimately safeguarding valuable assets like Crytocurrencys and other valuable assets.

# Our Dataset

This repository contains a script designed to generate text-based CAPTCHAs with randomly assigned colors & 5000 pre generated CAPTCHAS. The script allows for customization of font, spacing, and rotation parameters before generating each CAPTCHA. The dataset has been curated to exclude CAPTCHA instances that may be too difficult to recognize. For example, to avoid confusion between similarly-shaped characters (like the lowercase 'x' and uppercase 'X'), we have limited our font to 'Arial' the available character set to "023456789ABCDEFHJKLMNPQRTUVWXYabdefghikmnqrty". Additionally, we introduce noise in the form of salt-and-pepper noise and varying line thicknesses and lengths to further challenge recognition.

Somne Examples:

![image](https://github.com/user-attachments/assets/d1f0f462-302c-4604-801c-f49d74a64052)
![image](https://github.com/user-attachments/assets/90e97ff6-dee9-4393-86bc-e4ec53cbd463)
![image](https://github.com/user-attachments/assets/7b8f2865-5212-4749-b0c7-aecacc7d49ed)
![image](https://github.com/user-attachments/assets/2d5319e1-c43b-440a-a01c-0f9d0df931aa)
![image](https://github.com/user-attachments/assets/c3433463-dd11-43f0-9b2d-18cc877535a3)
![image](https://github.com/user-attachments/assets/41b720b4-acbc-4d31-96a1-17919e78cf19)
![image](https://github.com/user-attachments/assets/227cb327-3475-456d-9855-9cd87dc238dd)
![image](https://github.com/user-attachments/assets/f7d0024e-0c5b-4475-a66f-00909eab600f)
![image](https://github.com/user-attachments/assets/4a6e04b1-968f-4def-aafb-818f689b966e)




# Dataset Splitter
This repo also includes a short script wich splits the entire dataset in smaller sets for Training Validation and Testing. The Sizes of the sets can be adjusted in percent


# CNN
## Overview:
A CNN-based CAPTCHA solver is a deep learning model that uses Convolutional Neural Networks (CNNs) to recognize and decode CAPTCHA images by extracting visual patterns and classifying them into characters.

Building Blocks:
Convolutional Layer (Feature Extraction)
Purpose: Identifies spatial features such as edges, shapes, and textures within the CAPTCHA image.
How it works: Uses convolutional filters to detect patterns, followed by pooling layers to reduce dimensionality while preserving key features.
Role in CAPTCHA solving: Converts raw pixel data into a structured feature representation.

Fully Connected Layer (Classification):
Purpose: Interprets the features extracted by the CNN and assigns them to specific characters.
How it works: The high-level features are passed through dense layers, applying activation functions (e.g., Softmax) to classify each character.
Role in CAPTCHA solving: Maps the extracted features to corresponding alphanumeric outputs.

Decoding and Prediction:
The model predicts each character in the CAPTCHA and reconstructs the final text.

## Modell Aritecture:
Data Loading:
  Description: The dataset is loaded and split into training and test data.
  Split: 80% training dataset, 20% test/validation set.
  Additional Data Sources: Shared datasets can be used for better performance.
  
Preprocessing:
  Grayscale Conversion: All images are converted to grayscale.
  Rescaling: Images are resized to a uniform size of 25x67 pixels to ensure consistency in model input.
  
Model Architecture:
  Convolutional Layers:
    4 convolutional layers extract features and reduce image size.
    MaxPooling layers reduce dimensions while retaining important features.
    
  Dropout Layers:
    2 dropout layers help prevent overfitting.
    
  Dense Layer:
    A fully connected layer processes the extracted features for final classification.
    
Optimization:
  Hyperparameter Tuning: Adam optimizer is used for automatic learning rate adjustment.
  Loss Function: Designed to minimize the difference between predicted and actual values.
  
CNN Model Training:
  The model is trained using the processed dataset and optimized parameters.
  The weights of the layers are gradually improved to maximize recognition accuracy.
  
Evaluation and Output:
  Accuracy Calculation: The model is evaluated using the test set, measuring both individual character and full CAPTCHA accuracy.
  Visual Representation: Selected results are displayed graphically for better interpretation.


# CRNN
## Overview:
A CRNN is a combination of CNNs and RNNs that leverages the strengths of both models—CNNs for spatial feature extraction and RNNs for sequential modeling—making it ideal for tasks like speech and text recognition, where both local patterns and temporal dependencies are important.

Building Blocks:
Convolutional Layer (CNN part):

Purpose: Extracts spatial features from the input data (images, videos, or spectrograms).
How it works: CNNs use convolutional filters to capture local patterns (e.g., edges, textures) and produce feature maps. These layers are usually followed by pooling layers to reduce spatial dimensions.
Role in CRNN: The CNN layers act as feature extractors, transforming the raw input into a higher-level feature representation.
Recurrent Layer (RNN part):

Purpose: Models the temporal or sequential relationships between the features extracted by the CNN.
How it works: RNNs (or more commonly, LSTMs or GRUs) process sequences of data and maintain a hidden state that captures information over time.
Role in CRNN: The RNN layers capture the sequential dependencies between the features output by the CNN layers, allowing the model to process data in a time-dependent manner.
Fully Connected Layer (or Output Layer):

Purpose: Makes final predictions based on the features learned by the CNN and RNN layers.
How it works: After the sequential data is processed by the RNN, the output is passed through one or more fully connected layers to generate the final output, such as class probabilities in the case of classification tasks.
Role in CRNN: The fully connected layer helps in mapping the learned features to a final output space (e.g., words, characters, or class labels).

## Modell Aritecture:
Our model is designed for Optical Character Recognition (OCR) tasks, where the goal is to convert images of text into readable character sequences. We chose this architecture for its ability to efficiently handle the complexities of sequential text recognition, especially in cases where character positioning, rotation, and noise can vary. The combination of convolutional layers for feature extraction and recurrent layers (Bidirectional LSTM) for capturing sequential dependencies allows the model to effectively process images and learn the relationships between characters in a sequence.

## Architecture:

### Input Layer:
**Shape:** The model takes in images of size (image_height, image_width, 1) (grayscale images).
Preprocessing: The pixel values are scaled to be between 0 and 1 for better model performance.
Transpose Layer:

The image is transposed (the width and height are swapped) to prepare the data for the next layers.
### **Convolutional Layers:**

**Conv2D Layers:**

These layers detect basic features like edges, textures, and shapes in the image. The model uses three convolutional layers with 32, 64, and 128 filters, each followed by a MaxPooling2D layer to reduce the image size but keep important information.
The first two pooling layers reduce both height and width, and the third one reduces only the height.

**Reshaping and Dense Layer:**

After the convolutional layers, the feature maps are reshaped to fit the data into a dense layer, which processes the information as a whole.
A Dense Layer with 128 units and Dropout (to prevent overfitting) is applied.

### **Recurrent Layers:**

**Bidirectional LSTM:** 

This layer helps the model understand the sequence of data over time (like handwriting or speech). The bidirectional part means it looks at both past and future contexts in the sequence to make better predictions.

**Output Layer:**

The final layer uses a softmax activation function to output the probability distribution of each character (or class) in the sequence, based on the training data.
The number of units is the total number of possible characters + 1 (for the blank character used in CTC loss).
Custom Loss Function (CTC Loss):

The model uses CTC Loss (Connectionist Temporal Classification), which helps the model learn to output a sequence of characters even when the timing (or alignment) of the characters is unknown. This is crucial for OCR tasks.

**Model Compilation:**

The model is compiled with the Adam optimizer and the custom CTC loss function.

# Stregths:
-Rotation is not a problem

-Random positioning of characters is handled well

-Recognizes patterns in the dataset effectively

-Can predict characters that may not be easily distinguishable

# Weaknesses:
-CRNNs are resource-intensive, which can affect real-time performance

-Sourcing diverse and high-quality data for training can be challenging

-Preprocessing for CRNNs is generally more difficult compared to other models

-Limited ability to handle complex distortions, such as overlapping or warped characters

-Performance can be compromised under adversarial attacks designed to deceive the model

# Accuracy of our CRNN:
After training for 200 epochs with early stopping set to 20, the training was halted after 112 epochs due to the lack of significant improvements. Following the training, we evaluated the model’s performance on a separate test dataset containing 250 images. The model achieved a CTC loss as low as 0.0013 and a total accuracy of 97.2%, which is a pretty solid result! 
For evaluation, we compared the model's predictions, using the best weights, with the actual labels.

### Training Curve:
![gtihub buld](https://github.com/user-attachments/assets/00e7433b-38d8-4bbd-8d98-4db872f167cf)
### Accuracy:
![gihubbbild2](https://github.com/user-attachments/assets/2bef8fa8-7586-488f-8d22-765137f2d280)

# CRNN Conclusion
Achieving an accuracy of 97.2% is a remarkable result for our CRNN model, demonstrating its strong ability to recognize and predict text in images despite challenges like noise, rotation, and random character positioning. This high performance highlights the model's ability to leverage the strengths of both convolutional and recurrent layers—CNNs for effective feature extraction and RNNs (Bidirectional LSTM) for capturing temporal dependencies in the text sequences.

While CRNNs offer great potential, they also have their limitations. They are computationally intensive, which may affect real-time performance, and sourcing diverse, high-quality data for training can be challenging. Moreover, preprocessing is more complex for CRNNs compared to traditional models, and the model may struggle with highly distorted inputs or adversarial attacks.

Nevertheless, CRNNs remain a powerful tool for OCR tasks, offering significant advantages in processing and predicting text sequences in varied conditions. With continued improvements in training data quality, model architecture, and preprocessing techniques, CRNNs can be further enhanced to address their current weaknesses.

# LightGBM Model
LightGBM (Light Gradient Boosting Machine) is a powerful and efficient machine learning model designed for high performance and speed with large datasets. It is particularly well-suited for classification and regression tasks and has been adapted here specifically for captcha recognition.

## Data Ingestion
The first step in the process is data ingestion, where captcha images are split into their individual characters and stored explicitly. This separation is crucial to ensure that the model can accurately process and recognize each component of the captcha.

## Data Splitting
Once the data is ingested, it is divided into training, testing, and validation sets. Typically, 80% of the data is used for training, while the remaining 20% is split between test and validation sets. This division allows for comprehensive evaluation of the model's performance both during and after training.

## Preprocessing
Before the actual training, the data needs to be prepared. This step includes scaling pixel values, normalizing the datasets, and cleaning the data to remove any noise or distortions. Proper data preparation is key to improving the model's performance.

## Hyperparameter Optimization with Flaml
To achieve the best results, Flaml is used to find the optimal hyperparameters for the LightGBM model. Flaml is an automated tool for hyperparameter optimization that conducts an efficient search for the best parameters, further enhancing the model's performance.

## Machine Training with LightGBM
The prepared LightGBM model is trained on the training dataset using the optimized hyperparameters. This step involves fitting the model to the training data to learn patterns and relationships necessary for correctly recognizing captcha characters.

## Evaluation and Output
After training, the model is evaluated on the test dataset to verify its accuracy. The accuracy is calculated for both individual characters and the complete captcha. To demonstrate the model's performance and accuracy, results are visualized and compared with the actual labels.

## LightGBM Conclusion
The LightGBM model achieved an accuracy of approximately 80% on the captcha test set. While this is a commendable result, it highlights some limitations of using LightGBM for captcha recognition. The model excels at recognizing individual characters, but it is not fully optimized for solving complete captchas due to its primary focus on single character recognition. This constraint suggests that while LightGBM can be a useful tool for captcha recognition, it might require additional techniques or complementary models to handle the complexities of full captcha sequences effectively.


# Conclusion
Our analysis shows that while machines solve CAPTCHAs significantly faster than humans, they also make more mistakes. Through targeted model optimizations—especially in handling noise and different fonts—we were able to significantly improve recognition accuracy. Despite initial challenges, we achieved all our goals, and the final solution exceeded our expectations.

By implementing techniques such as Levenshtein distance similarity checks and dictionary matching, we further reduced the error rate. However, real-world applications still face challenges, particularly with web scraping due to additional security measures.

Our results confirm that text-based CAPTCHAs no longer provide sufficient security against AI-driven attacks. Therefore, we recommend switching to alternative CAPTCHA methods to ensure future protection.

# Credits
This project was made possible by the following open-source repositories:

https://github.com/EVOL-ution/Captcha-Recognition-using-CRNN?tab=MIT-1-ov-file

https://github.com/DrMahdiRezaei/Deep-CAPTCHA/tree/master

Minor parts of the work, including the splitting script debugging and optimizing tasks, were assisted by ChatGPT and Claude AI.

If you are the owner of any code used here and would like specific attribution, please let me know!
