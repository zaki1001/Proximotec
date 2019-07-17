# -*- coding: utf-8 -*-
"""
Created on Tue Apr 02 19:18:33 2019

@author: Yousuf
"""
'''
import gzip
import io
input = gzip.GzipFile("E:\\proximotex\\Information\\Signals\\p1.txt", 'rb',compresslevel=9)

kkk = gzip.GzipFile("E:\\proximotex\\Information\\Signals\\p1.txt", 'rb')
s = input.read()
#gzipfile = gzip.GzipFile('', 'rb', 9, BytesIO(s))
input.close()

output = open("E:\\proximotex\\Information\\Signals\\xxtest.csv", 'wb')
output.write(s)
output.close()
from io import BytesIO
print("done")
bindata = bytearray(s)
#buf = BytesIO(input)
#f = gzip.GzipFile(fileobj=buf)
#r = f.read()
with gzip.open("zen.txt.gz", "wb") as f:
	f.write(bindata)

'''

#feat=['magnitude_mean','z_mean','Moving_Average_max','Moving_Average_min','Moving_Average_mean','herm_fft_sum']
import os
import datetime
g="E:\\proximotex\\Signals\\Signals"
dirs=os.listdir( g )
d=[]
for files in dirs:
    print (files)
    k=files[-19:-4]
    d.append(k)
lk=[]
for ll in range(0,len(d)):
    lk.append(datetime.datetime.strptime(str(d[ll]),'%Y%m%d-%H%M%S'))
#kk=d[0].strftime('%Y%m%d-%h%M%S')
import numpy as np
import pandas as pd
feat=pd.DataFrame()
feat1=[]
import matplotlib.pyplot as plt
fd=[]
fuck=[]
import os
for i in range(0,len(dirs)):
        audio_path = g+"\\"'%s' %(dirs[i])
        date=lk[i]
        fp = open(audio_path,"rb")
        data = fp.read()
        bindata = bytearray(data)
        from scipy import signal
        z=list(bindata)
        z=np.asarray(z)
        sr=len(z)
    
        magnitude=np.abs(np.fft.rfft(z))
        freq=np.abs(np.fft.fftfreq(sr, 1.0/44100)[:sr//2+1])
        magnitude=magnitude[:sr//2+1]
        spectral_centroid=np.sum(magnitude*freq)/np.sum(magnitude)
        magnitude_mean=magnitude.mean()
        hermitianfft=np.fft.hfft(z)  
        herm_fft_mean=hermitianfft.mean()
        z_mean=z.mean()
        Moving_Average=np.convolve(z,np.ones((50, ))/50,mode='valid')
        Moving_Average_max=max(Moving_Average)
        Moving_Average_min=min(Moving_Average)
        Moving_Average_hfft=np.convolve(hermitianfft,np.ones((50, ))/50,mode='valid')
        f1,Psd=signal.welch(z,fs=44100,nperseg=1024)#power spectral density
        fin1=np.vstack((Psd,f1))
        fin1=fin1.transpose()
        Moving_Average_mean=Moving_Average.mean()
        Moving_Average_mean_hftt=Moving_Average_hfft.mean()
        fd.append(Psd)
        name=dirs[i][:-19]+str(date)
        feat2=[name,magnitude_mean,z_mean,Moving_Average_max,Moving_Average_min,Moving_Average_mean,Moving_Average_mean_hftt]
        #feat.append(feat1)
        #df={'Name':[dirs[i]],'magnitude_mean':[magnitude_mean],
                        #'z_mean':[z_mean],'Moving_Average_max':[Moving_Average_max],
                        #'Moving_Average_min':[Moving_Average_min],'herm_fft_sum':[herm_fft_sum]}
        feat1.append(feat2)
        #temp=pd.DataFrame(df)
        #d=pd.concat([feat,temp])
x=[]
final=pd.DataFrame(feat1,columns=['Name','magnitude_mean','z_mean','Moving_Average_max','Moving_Average_min','Moving_Average_mean','hftt_mean'])
#for l in range(0,len(fd)-1):
    #exec 'x%s = fd[%s]' %(l,l)
fd=np.asarray(fd)
final1=final.T
final2=pd.DataFrame(fd)

ff=len(dirs)
final2['Component']=np.asarray(dirs[0:ff])
final3=final2.T
frames=[final1,final3]
results=pd.concat(frames)
                        
''' 
import numpy as np
import pandas as pd
audio_path="E:\\proximotex\\Information\\Signals\\p1.txt"
fp = open(audio_path,"rb")
data = fp.read()
bindata = bytearray(data)
#with gzip.open("E:\\proximotex\\Information\\Signals\\p1.txt.gz", "wb") as f:
#	f.write(bindata)
#import struct
#testResult = struct.unpack('>HH', bindata)
from scipy import signal
z=list(bindata)
z=np.asarray(z)
sr=len(z)
    #choma_stft = librosa.feature.chroma_stft(y=z, sr=sr)
magnitude=np.abs(np.fft.rfft(z))
freq=np.abs(np.fft.fftfreq(sr, 1.0/44100)[:sr//2+1])
magnitude=magnitude[:sr//2+1]
spectral_centroid=np.sum(magnitude*freq)/np.sum(magnitude)   
#f,t,stft=signal.stft(z[:20000],fs=44100,nperseg=25)
magnitude_mean=magnitude.mean()
#stft_mean=[]
#for k in range(0,(len(stft[:,0])+1)):
    #try:
        #stft_mean.append(stft[:,k].mean())
    #except:
        #''''''
hermitianfft=np.fft.hfft(z)    
#herm_fft_mean=hermitianfft.mean()
z_mean=z.mean()
#stft_mean_mean=np.asarray(stft_mean).mean()
#stft_mean_mean_real=stft_mean_mean.real
#stft_mean_mean_imag=stft_mean_mean.imag
        
#def running_mean(z,N):
    #cumsum=np.cumsum(np.insert(z,0,0))
    #return (cumsum[N:] - cumsum[:-N]) /N
        #x=np.random.random(100000)
#N=1000
#D=running_mean(z,N)
Moving_Average=np.convolve(z,np.ones((50, ))/50,mode='valid')
Moving_Average_max=max(Moving_Average)
Moving_Average_min=min(Moving_Average)
Moving_Average_hfft=np.convolve(hermitianfft,np.ones((50, ))/50,mode='valid')
f1,Psd=signal.welch(z,fs=44100,nperseg=1024)#power spectral density
fin1=np.vstack((Psd,f1))
fin1=fin1.transpose()
Moving_Average_mean=Moving_Average.mean()
Moving_Average_mean_hftt=Moving_Average_hfft.mean()
#select=z[:20000]
'''









