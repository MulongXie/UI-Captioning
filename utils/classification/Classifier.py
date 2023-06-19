from tensorflow import keras
from keras_applications.resnet50 import ResNet50
from keras_applications import resnet50
from keras_applications.imagenet_utils import decode_predictions
from keras.utils import img_to_array
from keras.utils import load_img
import matplotlib.pyplot as plt
import PIL

import cv2
import numpy as np


class Classifier:
    def __init__(self, image_net_file='classification/data/imagenet1000.txt'):
        '''
        Use ImageNet pretrained model to classify content in image element
        '''
        self.image_net_file = image_net_file
        self.image_net_cls = [line.split(':')[-1][:-1].replace('\'', '').replace(',', '') for line in open(image_net_file, 'r')]
        self.resnet = ResNet50(backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        print('*** Load ResNet50 pretrained on ImageNet ***')

    def predict_img_files(self, img_files, show=False):
        '''
        Predict class for image files
        :param img_files: list of image file paths
        :param show: boolean
        '''
        images = []
        orgs = []
        for img_file in img_files:
            img = cv2.imread(img_file)
            orgs.append(img)
            images.append(img_to_array(cv2.resize(img, (224,224))))
        x = resnet50.preprocess_input(np.array(images), 'channels_last')
        predictions = self.resnet.predict(x)
        labels = [self.image_net_cls[np.argmax(pred)] for pred in predictions]
        if show:
            for i in range(len(orgs)):
                print(labels[i])
                cv2.imshow('img', orgs[i])
                key = cv2.waitKey()
                if key == ord('q'):
                    break
            cv2.destroyWindow('img')
        return labels

    def predict_images(self, images, show=False):
        '''
        Predict class for cv2 images
        :param images: list of cv2 images
        :param show: boolean
        '''
        images_proc = [img_to_array(cv2.resize(img, (224, 224))) for img in images]
        x = resnet50.preprocess_input(np.array(images_proc), 'channels_last')
        predictions = self.resnet.predict(x)
        labels = [self.image_net_cls[np.argmax(pred)] for pred in predictions]
        if show:
            for i in range(len(images)):
                print(labels[i])
                cv2.imshow('img', images[i])
                key = cv2.waitKey()
                if key == ord('q'):
                    break
            cv2.destroyWindow('img')
        return labels
