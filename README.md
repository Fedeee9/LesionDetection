# LesionDetection

<p align="center">
<img src="image/Principal_Image.png?raw=true"  width=500px />
</p>

LesionDetection is a simple neural network capable of identifying the lesions present in the human body through the use of bounding boxes.

## Getting Started

* Clone the GitHub repository
```
git clone https://github.com/Fedeee9/LesionDetection
```

* Install dependencies
```
sudo apt install python3-setuptools
sudo apt install python3-pip
sudo apt install python3-venv
```

* Create a virtual environment and install requirements modules
```
cd LesionDetection
python3 -m venv venv
source venv/bin/activate

python3 -m pip install "MODULE_NAME"
```

## Dataset
The split dataset with the related files can be found here: https://github.com/Fedeee9/Lesion_Dataset

## Running
* Training:
```
python3 train_bounding_box.py
```
* Evaluate:
```
python3 evaluate_bounding_box.py
```
* Test:
```
python3 predict_bounding_box.py --input "TEST_IMAGE_FILE"
```

## Result

### Examples

## Credits
* LesionDetection was developed by Federico Dal Monte
