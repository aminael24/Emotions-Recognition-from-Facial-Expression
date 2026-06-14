# Emotions Recognition from Facial Expression

## Project Description

This project is an **Emotion Recognition System from Facial Expressions** developed using **Computer Vision** and **Deep Learning**.

The main objective of this project is to detect a human face from an image or a real-time webcam stream, then classify the facial expression into one of several emotion categories.

This project was carried out as part of the **Vision System / Image Processing project** at **ENSA Tanger**.

**Supervisor:** LACHKAR ABDELMOUNAIM  
**School:** École Nationale des Sciences Appliquées de Tanger – ENSA Tanger  
**Module:** Vision System / Image Processing  

---

## Project Authors

This project was developed by:

- Amina El Haddad
- Tasnim Chaoui

---

## Overview

Facial emotion recognition is an important field of computer vision and artificial intelligence. It consists of analyzing facial expressions in order to recognize the emotional state of a person.

The system can recognize emotions such as happiness, sadness, anger, fear, surprise, disgust, and neutral expression.

This project includes the full pipeline:

- Dataset preparation
- Image preprocessing
- Data augmentation
- CNN model training
- Model evaluation
- Model conversion to TensorFlow Lite
- Real-time emotion detection using a webcam

---

## Dataset

The model was trained using the **FER2013 dataset**, a public dataset commonly used for facial expression recognition.

FER2013 contains grayscale facial images of size **48 × 48 pixels**, divided into seven emotion classes.

The dataset was loaded in the training notebook using the following paths:

```python
TRAIN_DIR = '/kaggle/input/datasets/msambare/fer2013/train'
TEST_DIR  = '/kaggle/input/datasets/msambare/fer2013/test'
```

The dataset is not included directly in this repository. It can be downloaded from Kaggle.

---

## Emotion Classes

The model is trained to classify facial expressions into the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## Project Structure

```text
Emotions-Recognition-from-Facial-Expression/
│
├── figures/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   └── sample_images.png
│
├── detector.py
├── emotion_model.keras
├── emotion_model.tflite
├── kaggle_training.ipynb
├── main.py
├── requirements.txt
├── speaker.py
└── README.md
```

---

## Repository Files Explanation

### `figures/`

This folder contains the figures generated during the training and evaluation phase.

It includes:

- `accuracy_curve.png`: shows the evolution of training and validation accuracy.
- `loss_curve.png`: shows the evolution of training and validation loss.
- `confusion_matrix.png`: shows the classification performance of the model.
- `class_distribution.png`: shows the distribution of images for each emotion class.
- `sample_images.png`: shows examples of images from the FER2013 dataset.

These figures are useful for analyzing and presenting the performance of the model.

---

### `requirements.txt`

This file contains the Python libraries required to run the project.

Main dependencies include:

```text
tensorflow
opencv-python
numpy
matplotlib
seaborn
scikit-learn
pyttsx3
Pillow
```

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

---

### `emotion_model.keras`

This file contains the trained deep learning model in Keras format.

It is the original CNN model trained on the FER2013 dataset. This model is mainly used for training, saving, evaluation, and possible future improvements.

---

### `emotion_model.tflite`

This file contains the **TensorFlow Lite version** of the trained model.

TensorFlow Lite is a lightweight format designed for faster inference and deployment on devices with limited resources, such as laptops without GPU or mobile devices.

In this project, the `.tflite` model is used by the real-time application because it is smaller and faster than the original `.keras` model.

In simple terms:

- `emotion_model.keras` = original trained model
- `emotion_model.tflite` = lightweight model used for fast real-time prediction

---

## What Does Lightweight Model Mean?

The term **lightweight model** means that the trained model was converted into a smaller and faster version.

The original model is saved as:

```text
emotion_model.keras
```

Then it is converted to TensorFlow Lite format:

```text
emotion_model.tflite
```

The `.tflite` model keeps the same goal as the original model: predicting emotions from facial expressions. However, it is more suitable for real-time use because it requires fewer resources and runs faster on CPU.

This is why the real-time application uses `emotion_model.tflite` instead of `emotion_model.keras`.

---

### `kaggle_training.ipynb`

This notebook contains the complete training process of the model.

It includes:

- Loading the FER2013 dataset
- Preprocessing facial images
- Data augmentation
- Building the CNN architecture
- Training the model
- Evaluating the model
- Generating performance figures
- Saving the model as `emotion_model.keras`
- Converting the model to `emotion_model.tflite`

---

### `detector.py`

This file contains the main emotion detection logic.

It is responsible for:

- Loading the TensorFlow Lite model
- Detecting faces using OpenCV
- Preprocessing detected faces
- Running emotion prediction
- Returning the predicted emotion and confidence score

---

### `main.py`

This is the main entry point of the real-time application.

It is used to:

- Open the webcam
- Read video frames
- Detect faces
- Predict emotions
- Display the detected emotion on the screen

---

### `speaker.py`

This file is used for audio feedback.

It uses text-to-speech to pronounce the detected emotion, making the application more interactive and accessible.

---

## Model Architecture

The chosen model is a **Convolutional Neural Network CNN** trained from scratch on FER2013 images.

The model was designed to work directly with grayscale images of size **48 × 48 pixels**.

The architecture is composed of several convolutional blocks followed by dense layers for classification.

The model includes:

- Convolutional layers
- Batch Normalization
- ReLU activation
- MaxPooling
- Dropout
- Dense layers
- Softmax output layer

The final layer contains **7 neurons**, corresponding to the seven emotion classes.

---

## Why a Custom CNN Was Used

At the beginning, transfer learning models such as MobileNetV2 and EfficientNetB0 were considered.

However, these models are usually trained on large RGB images from ImageNet, while FER2013 contains small grayscale facial images of size 48 × 48 pixels.

For this reason, a custom CNN trained from scratch was more suitable for this project.

---

## Training Environment

The training was performed on Kaggle using a GPU environment.

The training process used:

- TensorFlow / Keras
- FER2013 dataset
- Data augmentation
- Early stopping
- Learning rate reduction

The trained model was saved in Keras format as:

```text
emotion_model.keras
```

Then it was converted to TensorFlow Lite format as:

```text
emotion_model.tflite
```

---

## Data Preprocessing

Before training, the images were preprocessed as follows:

- Conversion to grayscale
- Resizing to 48 × 48 pixels
- Normalization of pixel values between 0 and 1
- Data augmentation applied to training images

Data augmentation techniques included:

- Horizontal flipping
- Rotation
- Zoom
- Width shift
- Height shift

These techniques help the model generalize better and reduce overfitting.

---

## Results

The model achieved a test accuracy of approximately:

```text
68.21%
```

This result is considered good for the FER2013 dataset, which is known to be difficult because of:

- Low image resolution
- Class imbalance
- Similarity between some emotions
- Variations in lighting, pose, and facial appearance

The best recognized emotions were generally:

- Happy
- Surprise

The most difficult emotion to recognize was:

- Fear

This is because fear can be visually confused with sadness or surprise.

---

## Real-Time Application

The real-time application uses the webcam to detect faces and predict emotions.

The processing pipeline is:

1. Capture a frame from the webcam.
2. Convert the frame to grayscale.
3. Detect faces using OpenCV Haar Cascade.
4. Crop the detected face.
5. Resize the face to 48 × 48 pixels.
6. Normalize the image.
7. Run prediction using the TensorFlow Lite model.
8. Display the predicted emotion and confidence score.

---

## CPU Optimization

The application was designed to run on a normal laptop without a dedicated GPU.

Several choices were made to improve performance:

- Use of TensorFlow Lite for faster inference
- Use of a lightweight model
- Face detection with OpenCV Haar Cascade
- Image resizing before detection
- Running prediction only on detected face regions

The TensorFlow Lite model allows the application to run smoothly in real time on CPU.

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/aminael24/Emotions-Recognition-from-Facial-Expression.git
```

### 2. Enter the project folder

```bash
cd Emotions-Recognition-from-Facial-Expression
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python main.py
```

or:

```bash
python detector.py
```

---

## How to Train the Model Again

To train the model again, open the notebook:

```text
kaggle_training.ipynb
```

The notebook contains all steps required to train the CNN model using the FER2013 dataset.

After training, the model is saved as:

```text
emotion_model.keras
```

Then it is converted to TensorFlow Lite format:

```text
emotion_model.tflite
```

---

## Technologies Used

- Python
- TensorFlow
- Keras
- TensorFlow Lite
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pyttsx3
- Pillow
- Kaggle
- FER2013 Dataset

---

## Academic Information

**Project Title:** Emotions Recognition from Facial Expression  
**Project Authors:** Amina El Haddad and Tasnim Chaoui  
**Module:** Vision System / Image Processing  
**School:** ENSA Tanger  
**Supervisor:** LACHKAR ABDELMOUNAIM  
**Dataset:** FER2013 from Kaggle  
**Model:** Custom CNN  
**Trained Model:** `emotion_model.keras`  
**Lightweight Model:** `emotion_model.tflite`  

---

## Conclusion

This project demonstrates how deep learning and computer vision can be used to recognize human emotions from facial expressions.

A custom CNN model was trained on the FER2013 dataset and then deployed in a real-time webcam application using TensorFlow Lite.

The project shows that separating the training phase from the deployment phase is an efficient solution:

- Training is performed on Kaggle using GPU.
- Real-time inference is performed locally on CPU using TensorFlow Lite.

This project was developed as part of the **Vision System / Image Processing project** at **ENSA Tanger**, under the supervision of **LACHKAR ABDELMOUNAIM**.

---

## License

This project is developed for academic purposes.
