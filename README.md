# Tensorflow Based Texture Synthesized Images Testing
This is the code to generate texture synthesized images and test Vgg net accuracy on those images.
The code is based on https://arxiv.org/pdf/1505.07376.pdf and https://github.com/meet-minimalist/Texture-Synthesis-Using-Convolutional-Neural-Networks

## Usage
To generate the texture synthesized images run the following command. Make sure you use tensorflow v1
```python
python3 Generating_images.py
```
To run the code testing the accuracy of prediction on synthesized images. Make sure you set the directory in the code
```python
python3 Testing.py
```
