import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io
import sys
import math
sys.path.append('../images/translate')
from images.translate import translate
from sklearn.model_selection import train_test_split
from skimage.transform import resize, rescale

def resize_image(img):
    new_size = [256, 256]
    print(img.shape)
    x = img.shape[1]    
    y = img.shape[0]

    if x==y:
        return resize(img, new_size)

    elif x > y:
        img = rescale(img, 256/x, anti_aliasing=True)
        gap_size = math.ceil(256 - (y * 256 / x)) // 2
        gap = np.zeros(shape=(gap_size, 256))

        # last resize for 255 or 257 values
        return resize(np.vstack((gap, img, gap)), new_size)
    else:
        gap_size = math.ceil(256 - (x * 256 / y)) // 2
        img = rescale(img, 256/y, anti_aliasing=True)
        gap = np.zeros(shape=(256, gap_size))

        return resize(np.hstack((gap, img, gap)), new_size)


def main():
    images = []
    categories = []
    rootdir = 'images/raw-img'
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            images.append(os.path.join(subdir, file))
            categories.append(translate[subdir.split("/")[-1]])

    df = pd.DataFrame(data= zip(images, categories), columns= ['file', 'category'])

    X = df['file'].values
    y = df['category'].values

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    # testing resize func
    img = resize_image(io.imread(x_train[11], as_gray=True))
    #img = io.imread(x_train[11], as_gray=True)
    io.imshow(img)
    plt.show()
    print(img.shape)


if __name__ == '__main__':
    main()


# numpy array normalization
# norm = np.linalg.norm(array)
# normal_array = array/norm
