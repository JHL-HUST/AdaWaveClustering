from AdaWave import *
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import csv

address = './syntheticData/waveData_'
# for noise in range(0,10):
noise = 5
filename = address+str(noise)+'.csv'
data = []
with open(filename) as f:
    f_csv = csv.reader(f)
    for row in f_csv:
         data.append(row)
data = np.array(data).astype(float)
# finished reading, start clustering
normData = normalizeData(data)
scale = 128
dim = 2
wavelet = 'db2'
wavelength = {'db1':0, 'db2':1, 'bior1.3':2}
dataDic = map2ScaleDomain(normData,scale)
dwtResult = ndWT(dataDic,2,scale,wavelet)
threshold = getThreshold(dwtResult)
print("threshold:")
print(threshold)

#show threshold on the chart
showThreshold(dwtResult,threshold)
lineLen = scale/2+wavelength.get(wavelet)
result = thresholding(dwtResult,threshold,lineLen,dim)
tags = markData(normData,result,lineLen)

#show the result after clustering
draw2Darray(normData[:,0],normData[:,1],np.array(tags))
quality = nmi(list(normData[:,normData.shape[1]-1]),tags)
print("AMI:")
print(quality)

