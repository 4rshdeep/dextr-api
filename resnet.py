from PIL import Image
import io
import requests
import cv2
import json
import torch
import torchvision
from torchvision import transforms, models
from torch.autograd import Variable

resnet = models.resnet101(pretrained=True)
resnet = resnet.cuda()

with open('labels.json') as data:
    data = json.load(data)
    labels = {int(key):value for (key, value)
          in data.items()}

resnet.eval()

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])


def get_prediction(imgpath):
    image = Image.open(imgpath)
    img_tensor = preprocess(image)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor).cuda()
    fc_out = resnet(img_variable)
    return labels[fc_out.cpu().data.numpy().argmax()]

def get_prediction_numpy(data):
    image = Image.fromarray(data, 'RGB')
    img_tensor = preprocess(image)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor).cuda()
    fc_out = resnet(img_variable)
    return labels[fc_out.cpu().data.numpy().argmax()]
