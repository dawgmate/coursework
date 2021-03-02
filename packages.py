import tensorflow as tf 
import numpy as np 
from PIL import Image
import keras
from tensorflow.python.keras.preprocessing import image as im
from tensorflow.python.keras import models
from tkinter.filedialog import askopenfilename
from tkinter import Tk

# names of layers
content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_cont_layers, num_style_layers = len(content_layers), len(style_layers)