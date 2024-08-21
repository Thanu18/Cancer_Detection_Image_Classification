# Cancer Detection Image Classification

This project is a Flask web application for classifying medical images into two categories: `HP` and `SSA`. The classification is performed using a pre-trained DenseNet model, RESNet50 and Fully connected convolution network. And the App is deployed in Aws.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [DataPreprocessing](#DataPreprocessing)
- [Model Explanation](#model-explanation)

## Project Overview

This project uses a DenseNet model, ResNet50 and FNN to classify images into two categories based on a medical dataset. And the best model is choosen for The classification model is served through a Flask web application that exposes a REST API for predictions.


## Installation

### Prerequisites

- Python 3.8+
- Docker (for containerized deployment)
- AWS Account: Ensure you have an AWS account.
- AWS CLI: Install and configure the AWS Command Line Interface (CLI).
- Elastic Beanstalk CLI (EB CLI): Install the Elastic Beanstalk Command Line Interface.


### Setting Up the Environment

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Thanu18/Cancer_Detection_Image_Classification.git
   cd cancer-detection-image-classification

2. ** Create the python virtual environment**
3. ** Install the dependencies in the requirements.txt**
   
   ```pip install -r requirements.txt

## Usage

1. Run locally with Docker
   After building teh Docker image, run the appliction locally:
   ```bash
   docker run -p 5000:5000 flask-image-classifier
2. Upload an Image
   - Open your web browser and go to http://localhost:5000.
   - Use the file input field to select an image file
   - Click the "Upload and predict" button to submit the image
3. View the prediction
   - After submission, the page will display the predicted class("HP" or "SSA")
4. Deploy the Application on Elastic Beanstalk
![Prediction](\pics\Prediction.png)


## Troubleshooting

   - Error: Could not import PIL Image : Endure the Pillow is installed
   - Deployment Issues: Verify Dockerfile and the AWS configuration
   - While preprocessing the data, heavily imbalanced data needed to flatted 
     the images to 1D array and use SMOTE analysis

## Data Preprocessing

### 1. Unzipping and Organizing Data
      The raw image data is initially stored in a zip file. We start by unzipping this file and organizing the images into directories for training and testing. This helps in structuring the data according to the required format for further processing.

### 2. Splitting Images into Training and Testing Sets
      The dataset is divided into training and testing subsets based on a CSV file containing image names and their corresponding labels. This step involves:
      Reading the CSV file and extracting image filenames and labels.
      Splitting the data into training and testing sets using a stratified approach to ensure that both sets have a balanced distribution of classes.
      Copying images to their respective directories for training and testing.

### 3. Loading and Preprocessing Images
      Images are then loaded and preprocessed to ensure consistency in size and format. This includes:
      Resizing images to a fixed size.
      Normalizing pixel values to be in the range [0, 1].
      Converting images to arrays suitable for model input.

### 4. Resampling using SMOTE
      To address class imbalance in the dataset, we use the Synthetic Minority Over-sampling Technique (SMOTE). SMOTE generates synthetic samples for the minority class to balance the class distribution. This step involves:
      Flattening image arrays to make them compatible with SMOTE.
      Applying SMOTE to create synthetic samples for the minority class.
      Reshaping the augmented dataset back to the original image shape.


## Model Explanation

This section provides a detailed explanation of the models used in this project.

## Basic CNN Model

### Architecture
- **Conv2D Layer**: 32 filters, (3, 3) kernel, ReLU activation
- **MaxPooling2D Layer**: (2, 2) pool size
- **BatchNormalization Layer**
- **Dense Layer**: 64 units, ReLU activation

### Compilation
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
![CNN_Model](\pics\CNN_Model.png)
![CNN_Accuracy](\pics\CNN_Accuracy.png)

### Training
- **Data Augmentation**: Applied during training.
- **EarlyStopping**: Stops training if validation accuracy does not improve for 5 epochs.
- **ModelCheckpoint**: Saves the best model based on validation accuracy.

## DenseNet201 Model

### Architecture
- **Base Model**: DenseNet201 (pre-trained on ImageNet)
- **Additional Layers**:
  - **GlobalAveragePooling2D**
  - **Dense Layer**: 1024 units, ReLU activation
  - **Dropout Layer**: 50% dropout rate

### Compilation
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
![Densenet](\pics\Densenet.png)


### Training
- **Data Augmentation**: Applied to training data.
- **EarlyStopping**: Stops training if validation loss does not improve for 3 epochs.
- **ModelCheckpoint**: Saves the best model based on validation loss.

## Model Deployment

### Predictive System
- **Purpose**: Classify new images using the trained models.
- **Procedure**:
  - Load and preprocess the image.
  - Use the trained model to predict the class.

### Saving and Loading Models
- **Model Saving**: Models are saved using `model.save()` and pickle for reuse.
- **Model Loading**: Models can be reloaded using `load_model()` or pickle.

## Example Usage

### Basic CNN Model

```python
# Load the model
model = load_model('/content/densenet_model.h5')

# Predict class for a new image
def predict_image(model_path, img_path):
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    return 'HP' if predictions[0][0] >= 0.5 else 'SSA'



   