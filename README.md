# CS 1430 Final Project
## Project Description
The goal of this project is to take in an image and read aloud an accurate description of that image. To achieve this, the image is evaluated by both a scene recognition convolutional neural network (CNN) and an image captioning CNN. The output from both models is combined into a single string, which is then read aloud using the [pyttsx3](https://pypi.org/project/pyttsx3/) library. The scene recognition CNN is trained using the 15 Scene dataset from homework 5. The image captioning CNN is trained with the Flickr 8K dataset. 
## Team Members
- Autumn Tilley
- Brennan Nugent
- Daniel Cho
- Olena Mursalova
## How to Run
### Dependencies
To run this project, make sure to use the CSCI 1430 python environment. Additionally, install the [pyttsx3](https://pypi.org/project/pyttsx3/) library using the following command:

```bash
pip install pyttsx3
```
### Training the Models
To train the two models, use the following commands from inside the project's code directory:
```bash
# To train the custom scene classification model
python run.py --task 1 --data ../data/15_Scene
# To train the custom image captioning model
python run.py --task 2 --data ../data/CIFAR-100-modified
# To train the VGG scene classification model
python run.py --task 3 --data ../data/15_Scene
# To train the VGG image captioning model
python run.py --task 4 --data ../data/CIFAR-100-modified
```
### Describing an Image
To hear a description of a particular image, run the following commands from inside the project's code directory:
```bash
# To evaluate the image using the custom models
python run.py --task 5 --data ../data/15_Scene --imagePath ../data/15_Scene/test/Office/image_0001.jpg
# To evaluate the image using the VGG models
python run.py --task 6 --data ../data/15_Scene --imagePath ../data/15_Scene/test/Office/image_0001.jpg
```
