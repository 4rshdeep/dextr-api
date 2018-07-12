# Deep Extreme Cut (DEXTR)

Api for [Deep Extreme Cut](https://arxiv.org/pdf/1711.09081.pdf) 
This paper explores the use of extreme points in an object (left-most, right-most, top, bottom pixels) as input to obtain precise object segmentation for images and videos. We do so by adding an extra channel to the image in the input of a convolutional neural network (CNN), which contains a Gaussian centered in each of the extreme points. The CNN learns to transform this information into a segmentation of an object that matches those extreme points. We demonstrate the usefulness of this approach for guided segmentation (grabcut-style), interactive segmentation, video object segmentation, and dense segmentation annotation. We show that we obtain the most precise results to date, also with less user input, in an extensive and varied selection of benchmarks and datasets.


## Installation 
* Clone the repo 
```
 git clone https://github.com/4rshdeep/dextr-api
```


* Install dependencies:
```
conda install pytorch torchvision -c pytorch
conda install matplotlib opencv pillow scikit-learn scikit-image
```
* Download the model by running the script inside models/:
```
cd models/
chmod +x download_dextr_model.sh
./download_dextr_model.sh
cd ..
```
* Run the demo from jupyter notebook
```
jupyter notebook
```
* Open the file demo.py 

## Functionality Implemented
1. Input: image and a set of points (x,y) coordinates for the points provided
2. Output: 
      * Boolean mask for the instance segmentation of the object within the points
      * Gives bounding box as part of the output
      * Also provides classification result of the returned object

Credits [DEXTR-Pytorch](https://github.com/scaelles/DEXTR-PyTorch/)
