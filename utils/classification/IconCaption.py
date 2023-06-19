import sys
import argparse, os, time, pickle, json, random, glob
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

import torch
from torchvision import transforms


class IconCaption:
    def __init__(self, vocab_path="/classification/model_results/vocab_idx2word.json",
                 model_path="/classification/model_results/labeldroid.pt"):
        self.vocab_path = vocab_path
        self.model_path = model_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = (224,224)
        self.img_transforms_test = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[ 0.485, 0.456, 0.406 ], std=[ 0.229, 0.224, 0.225 ])])
        self.model = None
        self.idx2word = None

        self.load_model()

    def load_model(self):
        # Load vocabulary idx2word dict
        with open(self.vocab_path, 'r') as f:
            # print(('[INFO] Loading idx2word %s' % self.vocab_path))
            self.idx2word = json.load(f)
        # Load model
        if os.path.exists(self.model_path):
            # print(('[INFO] Loading checkpoint %s' % self.model_path))
            self.model = torch.load(self.model_path)
            self.model.to(self.device)
            self.model.eval()
        else:
            print("[Error] the model path does not exist -", self.model_path)
            sys.exit(0)

    def predict_images(self, images):
        # convert cv2 image to PIL image
        if type(images[0]) == np.ndarray:
            images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
        sentences = []
        for img in images:
            img_torch = self.img_transforms_test(img)           # resize to [3, 244, 244]
            img_torch = img_torch.unsqueeze(0).to(self.device)  # convert to torch tensor and add one dimension -> [1, 3, 244, 244]
            word_ids = self.model(img_torch).cpu().numpy()[0]   # generate a [1,15] size list of ids to indicate words
            # Convert word_ids to words
            words = []  # words in the caption, mapped from word_ids
            for word_id in word_ids:
                word = self.idx2word[str(word_id)]
                if word == '<end>':
                    break
                words.append(word)
            sentence = ' '.join(words[1:])  # combine the words into a caption sentence
            sentences.append(sentence)
        # print(sentences)
        return sentences

    def predict_image_files(self, file_names):
        images = []
        for file in file_names:
            images.append(Image.open(file).convert('RGB'))
        self.predict_images(images)


if __name__ == '__main__':
    icon_cap = IconCaption()
    icon_cap.load_model()
    icon_cap.predict_image_files(['data/a1.jpg', 'data/a2.jpg'])
