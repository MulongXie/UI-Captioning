import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import time
import os
import json
from PIL import Image
import torch.multiprocessing as mp


class IconClassifier:
    def __init__(self, model_path='model_results/best-0.93.pt', class_path='model_results/iconModel_labels.json'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.model = torch.load(model_path, map_location=self.device).to(self.device)
        self.class_names = json.load(open(class_path, "r"))

    def predict_images(self, imgs):
        # convert cv2 image to PIL image
        if type(imgs[0]) == np.ndarray:
            imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in imgs]

        inputs = [self.transform_test(img) for img in imgs]
        inputs = torch.stack(inputs).to(self.device)
        # forward
        with torch.set_grad_enabled(False):
            outputs = self.model(inputs)
            outputs = nn.Softmax(dim=1)(outputs)
            values, preds = torch.max(outputs, 1)

        results = []
        for j in range(inputs.size()[0]):
            poss = values[j].item()
            if poss > 0.8:
                results.append([self.class_names[preds[j]], poss])
            else:
                results.append(["other", poss])
        return results


if __name__ == '__main__':
    classifier = IconClassifier()
    images = ['data/a1.jpg', 'data/a2.jpg', 'data/a3.jpg']
    # pil_img = [Image.open(img).convert('RGB') for img in images]
    pil_img = [cv2.imread(img) for img in images]
    result = classifier.predict_images(pil_img)
    print(result)

