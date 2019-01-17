import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from AdaWave import *
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

filename = './syntheticData/waveData_5.csv'
data = []
with open(filename) as f:
    f_csv = csv.reader(f)
    # headers = next(f_csv)
    for row in f_csv:
        data.append(row)
data = np.array(data).astype(float)

normData = normalizeData(data)
scale = 128
dim = 2
wavelet = 'db2'
wavelength = {'db1':0, 'db2':1, 'bior1.3':2}
dataDic = map2ScaleDomain(normData,scale)
dwtResult = ndWT(dataDic,2,scale,wavelet)
threshold = getThreshold(dwtResult)
lineLen = scale/2+wavelength.get(wavelet)
result = thresholding(dwtResult,threshold,lineLen,dim)
tags = markData(normData,result,lineLen)

filename = './syntheticData/waveTags0.5n.csv'
multiTags = []
with open(filename) as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        multiTags.append(row)
multiTags = np.array(multiTags).astype(int)
multiTags = np.concatenate((multiTags,np.array([tags]).T), axis=1)
# draw a picture for 2d dataset
x = normData[:,0]
y = normData[:,1]
fig = plt.figure()

ax = fig.add_subplot(2,2,1)
color = multiTags[:,0] / np.amax(multiTags[:,0])
rgb = plt.get_cmap('jet')(color)
ax.scatter(x,y,color = rgb)
plt.title('skinny-dip cluster')

bx = fig.add_subplot(2,2,2)
color = multiTags[:,1] / np.amax(multiTags[:,1])
rgb = plt.get_cmap('jet')(color)
bx.scatter(x,y,color = rgb)
plt.title('k-means cluster')

cx = fig.add_subplot(2,2,3)
color = multiTags[:,2] / np.amax(multiTags[:,2])
rgb = plt.get_cmap('jet')(color)
cx.scatter(x,y,color = rgb)
plt.title('dbscan cluster')

dx = fig.add_subplot(2,2,4)
color = multiTags[:,3] / np.amax(multiTags[:,3])
rgb = plt.get_cmap('jet')(color)
dx.scatter(x,y,color = rgb)
plt.title('our algorithm')

plt.show()

