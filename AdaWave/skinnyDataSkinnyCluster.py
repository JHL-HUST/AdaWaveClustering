import numpy as np
import AdaWave
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import csv

address = './syntheticData/skinnyDipData_'
# for noise in range(0,10):
noise = 5
filename = address+str(noise)+'.csv'
data = []
with open(filename) as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        data.append(row)
data = np.array(data).astype(float)
data = data[:, range(1, 4)]
wavelet = 'db2'
scale = 128
wavelength = {'db1': 0, 'db2': 1, 'bior1.3': 2}
lineLen = scale/2+wavelength.get(wavelet)
dim = data.shape[1]-1
normData = AdaWave.normalizeData(data)
dataDic = AdaWave.map2ScaleDomain(normData,scale)
dwtResult = AdaWave.ndWT(dataDic,dim,scale,wavelet)
threshold = AdaWave.getThreshold(dwtResult)
print("threshold:")
print(threshold)

#show threshold on the chart
AdaWave.showThreshold(dwtResult,threshold)
result = AdaWave.thresholding(dwtResult,threshold,lineLen,dim)
tags = AdaWave.markData(normData,result,lineLen)
lineLen = scale/2+wavelength.get(wavelet)
result = AdaWave.thresholding(dwtResult,threshold,lineLen,dim)
tags = AdaWave.markData(normData,result,lineLen)

#show the result after clustering
AdaWave.draw2Darray(normData[:,0],normData[:,1],np.array(tags))
quality = nmi(list(normData[:,normData.shape[1]-1]),tags)
print("AMI:")
print(quality)


