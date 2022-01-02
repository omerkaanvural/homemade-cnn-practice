import os
import numpy as np
import pandas as pd
from skimage import io
import sys
import math
import random
sys.path.append('../images/translate')
from images.translate import translate
from sklearn.model_selection import train_test_split
from skimage.transform import resize, rescale
from sklearn.preprocessing import LabelEncoder

def loss_function(output):
    pass

def softmax_activation(output):
    exp_values = np.exp(output)
    return exp_values / np.sum(exp_values)


def sigmoid_activation(input):
    return 1/(1 + np.exp(-input))


def normalize(img):
    norm = np.linalg.norm(img)
    return img/norm

def create_layer(neuron_size, input_size, output):
    weights = np.random.rand(input_size, neuron_size)
    biases = np.zeros(neuron_size)
    return np.dot(output, weights.T) + biases

def create_input_layer(img):
    return np.transpose(img.flatten())


def resize_image(img):
    new_size = [64, 64]
    x = img.shape[1]    
    y = img.shape[0]

    if x==y:
        return resize(img, new_size)

    elif x > y:
        img = rescale(img, 64/x, anti_aliasing=True)
        gap_size = math.ceil(64 - (y * 64 / x)) // 2
        gap = np.zeros(shape=(gap_size, 64))

        # last resize for 255 or 257 values
        return resize(np.vstack((gap, img, gap)), new_size)
    else:
        gap_size = math.ceil(64 - (x * 64 / y)) // 2
        img = rescale(img, 64/y, anti_aliasing=True)
        gap = np.zeros(shape=(64, gap_size))

        return resize(np.hstack((gap, img, gap)), new_size)


def main():
    images = []
    categories = []
    rootdir = 'images/raw-img'

    for subdir, dirs, files in os.walk(rootdir):
        category = subdir.split("/")[-1]
        if category == "raw-img":
            continue
        else:
            for file in files:
                images.append(os.path.join(subdir, file))
                categories.append(translate[category])

    df = pd.DataFrame(data= zip(images, categories), columns= ['file', 'category'])
    label_encoder = LabelEncoder()
    # alphabetically numerized
    df['category'] = label_encoder.fit_transform(df['category'])


    X = df['file'].values
    y = df['category'].values

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    # sample data to access some values about data itself
    img = resize_image(io.imread(x_train[10], as_gray=True))
    input_layer = create_input_layer(img)

    # for every ouput we have seperate bias and weight
    bias1, weight1 = random.random(), np.random.rand(len(input_layer))
    bias2, weight2 = random.random(), np.random.rand(len(input_layer))
    bias3, weight3 = random.random(), np.random.rand(len(input_layer))
    bias4, weight4 = random.random(), np.random.rand(len(input_layer))
    bias5, weight5 = random.random(), np.random.rand(len(input_layer))
    bias6, weight6 = random.random(), np.random.rand(len(input_layer))
    bias7, weight7 = random.random(), np.random.rand(len(input_layer))
    bias8, weight8 = random.random(), np.random.rand(len(input_layer))
    bias9, weight9 = random.random(), np.random.rand(len(input_layer))
    bias10, weight10 = random.random(), np.random.rand(len(input_layer))
    biases = np.array([bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8, bias9, bias10])
    weights = np.array([weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9, weight10])

    outputs = np.zeros(len(x_train))
    for i in range(len(x_train)):
        image = x_train[i]
        img = resize_image(io.imread(image, as_gray=True))
        input_layer = create_input_layer(img)
        output = normalize(np.dot(input_layer, weights.T) + biases)
        output = create_layer(10, len(output), output)
        output = create_layer(10, len(output), output)
        output = softmax_activation(output)




        print(f"predicted: {np.argmax(output)}, actual: {y_train[i]}")


if __name__ == '__main__':
    main()

