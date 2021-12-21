import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io


source = []
rootdir = 'images/raw-img'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        img = io.imread(os.path.join(subdir, file), as_gray=True)
        img_resized = np.resize(img, [256, 256])
        labeled_img = (img_resized, subdir.split("/")[-1])
        source.append(labeled_img)

print(len(source))











# plt.figure()
# plt.imshow(sample_image) 
# plt.show()