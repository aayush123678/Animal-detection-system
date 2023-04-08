from flask import Flask, render_template, url_for, request, redirect
from detecto import core, utils, visualize
import os
from pychatgpt import Chat, Options
from torchvision import models
import torch
import wikipedia
from PIL import Image
from werkzeug.utils import secure_filename
from torchvision import transforms
transform = transforms.Compose([  # [1]
    transforms.Resize(256),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
    transforms.Normalize(  # [5]
        mean=[0.485, 0.456, 0.406],  # [6]
        std=[0.229, 0.224, 0.225]  # [7]
    )])
dir(models)
ALLOWED_EXTENSIONS = {'png', 'jfif', 'mov', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)


@app.route('/')
def game1():
    return render_template('home.html')


@app.route('/document')
def game():
    return render_template('index.html')


@app.route('/implement')
def gam():
    return render_template('base.html')


def remove_repetitions(text):
    first_ocurrences = []
    for sentence in text.split("."):
        if sentence not in first_ocurrences:
            first_ocurrences.append(sentence)
    return '.'.join(first_ocurrences)


def trim_last_sentence(text):
    return text[:text.rfind(".")+1]


def clean_txt(text):
    return trim_last_sentence(remove_repetitions(text))


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, secure_filename(f.filename))

        f.save(file_path)
        model = models.alexnet(pretrained=True)
        img = Image.open(file_path)
        img_t = transform(img)

        batch_t = torch.unsqueeze(img_t, 0)
        model.eval()
        out = model(batch_t)
        _, index = torch.max(out, 1)
        with open('C:/Users/gauri/Downloads/Detecto Animal detection-20230408T151806Z-001/Detecto Animal detection/imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]

        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        predictions = [(classes[idx], percentage[idx].item())
                       for idx in indices[0][:5]]
        labeled, scores = predictions[0]
        print(labeled, scores)
        animal_name, sci_name = labeled.split(',')
        print(animal_name.capitalize())
        wiki_desc = wikipedia.summary(
            animal_name.capitalize(), sentences=4, auto_suggest=False)
        return render_template("base.html", creature_name=animal_name.capitalize(), creature_score=scores,  creature_desc=wiki_desc)
        final = []


if __name__ == '__main__':
    app.run(debug=True)
