#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:58:39 2020
@author: miagiandinoto
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import math
import wfdb
import pywt

startSamp = 110000
endSamp = 120000
N = endSamp - startSamp

os.chdir('/Users/dhruvmodi/Desktop/ECG-Filtering/mit-database/')
sig, fields = wfdb.rdsamp('118e06', channels=[1], sampfrom=startSamp, sampto=endSamp)
sig2 = sig.flatten()
fs = fields.get('fs')

# next line figures out level of decomposition so that lowest approximation
# coefficient takes in frequencies that are ,<=0.5 Hz (usual cutoff for removing baseline wander)
level = math.floor(math.log(fs / 0.5, 2))
waveName = 'db6'
coeffs = pywt.wavedec(sig2, waveName, level=level)

# plots original signal
fig = plt.figure(figsize=(20, 15))
fig.add_subplot(level + 2, 1, 1)
plt.plot(sig)
# plots all coefficients below original signal
for i in range(level):
    fig.add_subplot(level + 2, 1, i + 2)
    plt.plot(coeffs[i])
plt.show()
# reconstructing the signal with noise removed
# cA4=pywt.downcoef('a',sig2,wavelet='db6',level=4)
cD1 = np.array(coeffs[level])
abs_cD1 = np.absolute(cD1)
# THRESHOLD SELECTION
mediancD1 = np.median(abs_cD1)
sigma = mediancD1 / .6457
t = sigma * math.sqrt(2 * math.log(N, 10))

# thresholds coefficients- need to explore this more
newCoeffs = [(pywt.threshold(subArr, t, mode='soft')) for subArr in coeffs]

# this section replaces cA8(lowest approx. coeff that contains baseline wander)
# and cD1(contains most of the high frequency noise) with arrays of zeros
cA = coeffs[]
cD1 = coeffs[-1]
zerosA = np.zeros(cA.shape, dtype=float)
zerosD = np.zeros(cD1.shape, dtype=float)
newCoeffs[0] = zerosA
newCoeffs[-1] = zerosD

# reconstructs denoised signal without cA8 or cD1, with all other coeffs thresholded
denoised = pywt.waverec(newCoeffs, wavelet=waveName)

# plots original signal(top),denoised signal(middle), and clean signal(bottom)
fig2 = plt.figure(figsize=(20, 10))

fig2.add_subplot(3, 1, 1)
plt.plot(sig2)

fig2.add_subplot(3, 1, 2)
plt.plot(denoised)

fig2.add_subplot(3, 1, 3)

cleanSig, field = wfdb.rdsamp('118', channels=[1], sampfrom=startSamp, sampto=endSamp)
plt.plot(cleanSig.flatten())
plt.show()

plt.plot(sig)
plt.plot(denoised)
plt.show()
noise = sig2 - cleanSig
denoised = denoised
ms1 = np.mean(denoised**2)
ms2 = np.mean(noise**2)
SNR3 = 10 * np.log(ms1/ms2) #one method of SNR calculation in decibels
print("SNR: " + str(SNR3))