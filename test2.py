from IPython.display import display
from scipy.signal import butter, lfilter, filtfilt, freqz
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import os
import pandas as pd
import librosa
import shutil
import posixpath
from scipy.fft import fft, ifft
import wfdb
import cmath
from scipy.signal import *
from ltiarithmetic import TransferFunction, Z_R, Z_C, Z_L, s

if "record" is not locals():
    os.chdir("/Users/dhruvmodi/Desktop/ECG-Filtering/ecg-database/Person_01/")
#just some data reading
record = wfdb.rdrecord('rec_1')
wfdb.plot_wfdb(record=record, title='MIT record 1')
display(record.__dict__)

#read in noisy data for 2 seconds
signals, fields = wfdb.rdsamp('rec_1', channels=[0], sampfrom=0, sampto= 1000) #194400)
#read in clean data for 2 seconds
signals0, fields0 = wfdb.rdsamp('rec_1', channels=[1], sampfrom=0, sampto= 1000) #194400)
#assign variables and create time-frame
sampling_freq = 500
sampling_duration = 2
number_of_samples = int(sampling_freq * sampling_duration)
time = np.linspace(0, sampling_duration, number_of_samples, endpoint=False)


#plot noisy data vs clean
plt.plot(time, signals, 'b-', label='signal')
plt.plot(time, signals0, 'g-', linewidth=2, label='filtered signal')
plt.show()
#plot noise data vs clean (it's different)
noise = signals-signals0 #unfiltered - filtered should just leave noise
plt.plot(time, noise, 'k-')
plt.plot(time, signals0, 'g-', linewidth=2, label='filtered signal')
plt.show()


#do and plot filter
N  = 3 # Filter order
Wn = 0.1 # Cutoff frequency bw 0 & 1 (50 Hz cutoff bcz of fourier which is 500(fs) * 0.1)

b, a = signal.butter(N, Wn, 'low') #point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”)
fsignals = signal.filtfilt(b, a, signals, axis=0)

plt.plot(time, signals, 'b-', label='signal')
plt.plot(time, fsignals-0.05, 'g-', linewidth=2, label='filtered signal') #subtracting just for visuals
plt.show()

#potential SNR calculation
ms1 = np.mean(fsignals**2)
ms2 = np.mean(noise**2)
SNR = 10 * np.log(ms1/ms2) #one method of SNR calculation in decibels
print(SNR)

#transfer function; 2nd order RC filter using 1k resistors and 4u capacitors (look at bode plots)
#http://sim.okawa-denshi.jp/en/CRCRtool.php
'''
from ltiarithmetic import TransferFunction, Z_R, Z_C, Z_L, s

#cutoff Freq = (1/2piRC)
Z_1 = Z_R(800.)
Z_2 = Z_C(4E-6)
H = Z_2 / (Z_1 + Z_2)
'''

sys = signal.lti(1,[800*4E-6,1]) #cutoff freq of 49.7 (abt 50)
t = np.linspace(0, 2, 1000)
v_in = signals
tout, y, x = signal.lsim(sys, v_in, t)

plt.plot(t, signals)
plt.plot(t, y)
plt.show()

## Fourier Transform
signals = signals.T
f = fft(signals)
f = np.abs(f)
freq = np.fft.fftfreq(1000, d=1/500)
plt.plot(freq, f.T)
plt.show()

#from DWT_denoising import DWT_denoising -> for wavelet