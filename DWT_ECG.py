#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:59:17 2020

@author: miagiandinoto
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import wfdb
import pywt
import scipy.signal
import numpy.ma as ma

def r_isolate_wavelet(sig,fs,sig_len):
    level = math.floor(math.log(fs / 0.5, 2))
    waveName = 'db6'
    coeffs = pywt.wavedec(sig, waveName, level=level)
    
    cD1 = np.array(coeffs[level])
    # THRESHOLD SELECTION
    abs_cD1=np.absolute(cD1)
    mediancD1 = np.median(abs_cD1)
    sigma = mediancD1 / .6457
    t = sigma * math.sqrt(2 * math.log(sig_len, 10))

    coeffs2=pywt.wavedec(sig,wavelet=waveName,level=5)
    coeffs2= [(pywt.threshold(subArr, t, mode='soft')) for subArr in coeffs2]
    coeffs2[0]=np.zeros_like(coeffs2[0])
    coeffs2[-1]=np.zeros_like(coeffs2[-1])
    coeffs2[-2]=np.zeros_like(coeffs2[-2])
    
    
    r_wavelet=pywt.waverec(coeffs2,wavelet=waveName)
    return r_wavelet
    
def DWT_denoise(sig,fs,N):
    #next line figures out level of decomposition so that lowest approximation 
    #coefficient takes in frequencies that are ,<=0.05 Hz (usual cutoff for removing baseline wander)
    level=math.floor(math.log(fs/0.5,2))
    waveName='db6'
    coeffs= pywt.wavedec(sig, waveName,level=level)
       
    #reconstructing the signal with noise removed
    #cA4=pywt.downcoef('a',sig2,wavelet='db6',level=4)
    cD1 = np.array(coeffs[level])
    abs_cD1=np.absolute(cD1)
    # THRESHOLD SELECTION
    mediancD1 = np.median(abs_cD1)
    sigma = mediancD1 / .6457
    t = sigma * math.sqrt(2 * math.log(N, 10))
    
    #thresholds coefficients- need to explore this more
    newCoeffs = [(pywt.threshold(subArr,t,mode='soft')) for subArr in coeffs]
    
    #this section replaces cA8(lowest approx. coeff that contains baseline wander)
    #and cD1(contains most of the high frequency noise) with arrays of zeros
    cA=coeffs[0]
    cD1=coeffs[-1]
    zerosA=np.zeros(cA.shape,dtype=float)
    zerosD=np.zeros(cD1.shape,dtype=float)
    newCoeffs[0]=zerosA
    newCoeffs[-1]=zerosD
    
    #reconstructs denoised signal without cA8 or cD1, with all other coeffs thresholded
    denoised = pywt.waverec(newCoeffs,wavelet=waveName)
    return denoised