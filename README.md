# MLFFEViT
Code for Deepfake Face Detection via Discrete Wavelet Transform and Vision Transformer.
## Project Name and Introduction
We propose a Deepfake Face Detection via Discrete Wavelet Transform and Vision Transformer for enhancing the face tampering detection performance.

## Requirements

To run this project, you'll need to have the following libraries installed:

- Python 3.9+
- PyTorch
- torchvision
- matplotlib
- pytorch_wavelets
- PyWavelets
- facenet_pytorch
- sklearn.metrics

You can install the required libraries using pip:

```bash
pip install torch, torchvision, pytorch_wavelets,PyWavelets, matplotlib,facenet_pytorch,sklearn.metrics
```
The dataset can be downloaded from the links provided below.
1. FaceForensics++: https://github.com/ondyari/FaceForensics
2. Celeb-DF: https://www.kaggle.com/datasets/reubensuju/celeb-df-v2
3. Deepfake Image Dataset(DFID) : https://github.com/DariusAf/MesoNet 
4. We have created our own private dataset that is not available as a public dataset.

## Dataset Preparation

### Step 1: Convert Videos into Frames
You need to extract frames from videos using Preprocessing script and remove frames that contained incomplete or partial faces, hands, actors' clothing, and other items in the frames background.  

### Step 2: Dataset Structure

Organize your dataset in the following structure:

```
video_dataset/
    videos/
	Frames/
        face_image_dataset_split/
            train/
                class1/
                class2/
            val/
                class1/
                class2/
            test/
                class1/
                class2/
```


### Step 3: Split face_image_dataset into Train and Validation and Test Sets

Create your dataset according to the above structure.

## Training and testing the Model

Ensure that the paths to your dataset are correctly set.

### Train and Evaluation

Train your model and save the best model with minmum validation loss. You can load this model for further evaluation or inference.

## Results Visualization

The script includes plotting of training and validation accuracy, loss, and the confusion matrix.

### Loss and Accuracy Plots

After training, the script generates plots for training and validation loss and accuracy.

## Conclusion

This project demonstrates a complete workflow for training and evaluating a deepfake face detection model via discrete wavelet transform and the vision transformer.The provided details includes data loading, model training, evaluation, and visualization of results.
