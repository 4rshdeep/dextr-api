import os
import torch
from collections import OrderedDict
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from torch.nn.functional import upsample
import networks.deeplab_resnet as resnet
from mypath import Path
from dataloaders import helpers as helpers
import requests
import io
from resnet import get_prediction_numpy

modelName = 'dextr_pascal-sbd'
pad = 50
thres = 0.8
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

#  Create the network and load the weights
net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
# print("Initializing weights from: {}".format(os.path.join(Path.models_dir(), modelName + '.pth')))
state_dict_checkpoint = torch.load(os.path.join(Path.models_dir(), modelName + '.pth'),
                                   map_location=lambda storage, loc: storage)
# Remove the prefix .module from the model when it is trained using DataParallel
if 'module.' in list(state_dict_checkpoint.keys())[0]:
    new_state_dict = OrderedDict()
    for k, v in state_dict_checkpoint.items():
        name = k[7:]  # remove `module.` from multi-gpu training
        new_state_dict[name] = v
else:
    new_state_dict = state_dict_checkpoint
net.load_state_dict(new_state_dict)
net.eval()
net.to(device)
    
IMG_URL     = 'https://raw.githubusercontent.com/scaelles/DEXTR-PyTorch/master/ims/dog-cat.jpg'
EXTREME_PTS = [[93, 24], [334, 165], [60, 364], [0, 273]]


def dextr_helper(img_url=IMG_URL, extreme_pts=EXTREME_PTS):
    """
    @params 
    img_url - string containing url to the image
    extreme_pts - list of (x, y) extreme coordinate tuples 

    @returns tuple - (bbox, mask, pred)
    bbox (x_min, y_min, x_max, y_max) is the bounding box generated from he extreme points
    mask is a boolean numpy array indicating presence of instance 
    pred is the classification result 
    """

    response = requests.get(img_url)
    image = np.array(Image.open(io.BytesIO(response.content)))

    with torch.no_grad():
        extreme_points_ori = np.array(extreme_pts).astype(np.int)

        #  Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

        class_prediction = get_prediction_numpy(crop_image)
        # print("Class Prediction is : {}".format(class_prediction))

        # this is the bounding box to return (with 0 padding)
        actual_bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=0, zero_pad=True)

        #  Generate extreme point heat map normalized to image values
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [pad,
                                                                                                                      pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
        inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

        # Run a forward pass
        inputs = inputs.to(device)
        outputs = net.forward(inputs)
        outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
        outputs = outputs.to(torch.device('cpu'))

        pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)

        # Here result is of the shape of image, where True implies that part should be in the segment
        result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres

        return (actual_bbox, result, class_prediction)
