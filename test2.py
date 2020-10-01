from IPython.display import display
from scipy.signal import butter, lfilter, filtfilt, freqz
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import os
from scipy.fft import fft, ifft
import wfdb

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
N  = 2 # Filter order
Wn = 0.06 # Cutoff frequency bw 0 & 1 (30 Hz cutoff bcz of fourier which is 500(fs) * 0.06)
b, a = signal.butter(N, Wn, 'low') #point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”)
fsignals = signal.filtfilt(b, a, signals, axis=0)
plt.plot(time, signals, label='signal')
plt.plot(time, fsignals, linewidth=2, label='filtered signal') #subtracting just for visuals
plt.show()


#potential SNR calculation
ms1 = np.mean(fsignals**2)
ms2 = np.mean(noise**2)
SNR = 10 * np.log(ms1/ms2) #one method of SNR calculation in decibels
print("2nd Order Butter SNR: " + str(SNR))


## Fourier Transform
signals = signals.T
f = fft(signals)
f = np.abs(f)
freq = np.fft.fftfreq(1000, d=1/500)
plt.plot(freq, f.T)
plt.plot(np.linspace(0, 30,30), np.linspace(0, 30,30)) #check a good cutoff freq
plt.show()
signals = signals.T


#transfer function; 2nd order RC filter using 1k resistors and 4u capacitors (look at bode plots)
#http://sim.okawa-denshi.jp/en/CRCRtool.php

sys = signal.lti([36982.24852071],[1,576.92307692308,36982.24852071]) #cutoff freq of 50Hz
#sys = signal.TransferFunction([36982.24852071],[1,576.92307692308,36982.24852071]) #cutoff freq 30Hz

#Transfer function does same -> function is 36982.24852071 / s^2 + 576.92307692308s + 36982.24852071
#^ created from 2 Resistors at 1.3 kilo Ohms and 2 Caps at 4 micro Farrads

t = np.linspace(0, 2, 1000)
tout, y, x = signal.lsim2(sys, signals, t)

plt.plot(t, signals)
plt.plot(t, y)
plt.show()


#potential SNR calculation
ms1 = np.mean(y**2)
ms2 = np.mean(noise**2)
SNR2 = 10 * np.log(ms1/ms2) #one method of SNR calculation in decibels
print("2nd Order RC SNR: " + str(SNR2))


#R-Peak Detection

x1 = fsignals.T
x1 = x1.flatten()
x2 = y.T
x2 = x2.flatten()

peaks1, _ = signal.find_peaks(x1, distance=300)
np.diff(peaks1)
peaks2, _ = signal.find_peaks(x2, distance=300)
np.diff(peaks2)

plt.plot(x1)
plt.plot(peaks1, x1[peaks1], "x")
plt.show()
plt.plot(x2)
plt.plot(peaks2, x2[peaks2], "x")
plt.show()

#from DWT_denoising import DWT_denoising -> for wavelet