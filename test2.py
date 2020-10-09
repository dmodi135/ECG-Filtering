from IPython.display import display
from scipy.signal import butter, lfilter, filtfilt, freqz
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import os
from scipy.fft import fft, ifft
import wfdb
from ecgdetectors import Detectors
from DWT_ECG import DWT_denoise, r_isolate_wavelet

os.chdir("/Users/dhruvmodi/Desktop/ECG-Filtering/mit-database/")

sampTime = 30
sampling_freq = 360
sampTo = sampTime*sampling_freq

#read in noisy data for 2 seconds
signals, fields = wfdb.rdsamp('118e06', channels=[0], sampfrom=108000, sampto=108000 + sampTo) #194400)
#read in clean data for 2 seconds
signals0, fields0 = wfdb.rdsamp('118', channels=[0], sampfrom=108000, sampto=108000 + sampTo) #194400)
#assign variables and create time-frame
sampling_duration = sampTime
number_of_samples = int(sampling_freq * sampling_duration)
time = np.linspace(0, sampling_duration, number_of_samples, endpoint=False)

os.chdir("/Users/dhruvmodi/Desktop/ECG-Filtering/")
signals = signals-np.mean(signals)
signals0 = signals0-np.mean(signals0)
'''
difference1 = 0-signals[0]
signals = signals + difference1
difference2 = 0-signals0[0]
signals0 = signals0 + difference2
'''

#plot noisy data vs clean
plt.plot(time, signals, 'b-', label='signal')
plt.plot(time, signals0, 'g-', label='filtered signal')
plt.legend(['Unfiltered ECG', 'True ECG'])
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Unfiltered vs. True ECG')
plt.savefig('Plots/Original ECG vs Noisy ECG.png')
plt.show()
#plot noise data vs clean (it's different)
noise = signals-signals0 #unfiltered - filtered should just leave noise
plt.plot(time, noise, 'k-')
plt.plot(time, signals0, 'g-')
plt.legend(['Unfiltered ECG', 'Pure Noise'])
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Unfiltered vs. Noise')
plt.savefig('Plots/Noisy.png')
plt.show()


#do and plot filter
N  = 2 # Filter order
Wn = 0.06 # Cutoff frequency bw 0 & 1 (30 Hz cutoff bcz of fourier which is 500(fs) * 0.06)
b, a = signal.butter(N, Wn, 'low') #point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”)
fsignals = signal.lfilter(b, a, signals, axis=0)
plt.plot(time, signals)
plt.plot(time, fsignals) #subtracting just for visuals
plt.legend(['Unfiltered ECG', 'Butter Filtered ECG'])
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Butter Filtered Data')
plt.savefig('Plots/2nd Order Butter @ 30.png')
plt.show()
#potential SNR calculation
ms1 = np.mean(fsignals**2)
ms2 = np.mean((fsignals-signals0)**2)
SNR = 10 * np.log(ms1/ms2) #one method of SNR calculation in decibels
print("2nd Order Butter SNR: " + str(SNR))


## Fourier Transform
signals = signals.T
f = fft(signals)
f = np.abs(f)
freq = np.fft.fftfreq(sampTo, d=1/sampling_freq)
plt.plot(freq, f.T)
#plt.plot(np.linspace(0, 30,30), np.linspace(0, 30,30)) #check a good cutoff freq
plt.legend(['Frequency Magnitudes'])
plt.grid(False)
plt.xlabel('Frequency')
plt.ylabel('mV')
plt.title('Fourier Transformed Data')
plt.savefig('Plots/Fourier.png')
plt.show()
signals = signals.T


#transfer function; 2nd order RC filter using 1.33k resistors and 4u capacitors (look at bode plots)
#http://sim.okawa-denshi.jp/en/CRCRtool.php

sys = signal.lti([35332.69263384],[1,563.90977443609,35332.69263384]) #cutoff freq of 50Hz
#sys = signal.TransferFunction([35332.69263384],[1,563.90977443609,35332.69263384]) #cutoff freq 30Hz

#Transfer function does same -> function is 35332.69263384 / s^2 + 563.90977443609s + 35332.69263384
#^ created from 2 Resistors at 1.33 kilo Ohms and 2 Caps at 4 micro Farrads

t = np.linspace(0, sampTime, sampTo)
tout, y, x = signal.lsim2(sys, signals, t)

plt.plot(t, signals)
plt.plot(t, y)
plt.legend(['Unfiltered ECG', 'RC Filtered ECG'])
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('RC Filtered Data')
plt.savefig('Plots/2nd Order RC @ 30.png')
plt.show()
#potential SNR calculation
ms1 = np.mean(y**2)
ms2 = np.mean((y-signals0)**2)
SNR2 = 10 * np.log(ms1/ms2) #one method of SNR calculation in decibels
print("2nd Order RC SNR: " + str(SNR2))


#Wavelet Filter
s1= signals.flatten()
signalsW = DWT_denoise(s1, sampling_freq, sampTo)

plt.plot(t, signals)
plt.plot(t, signalsW)
plt.legend(['Unfiltered ECG', 'Wavelet Filtered ECG'])
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Wavelet Filtered Data')
plt.savefig('Plots/Wavelet Filtered.png')
plt.show()
#potential SNR calculation
ms1 = np.mean(signalsW**2)
ms2 = np.mean((signalsW-signals0)**2)
SNR3 = 10 * np.log(ms1/ms2) #one method of SNR calculation in decibels
print("Wavelet SNR: " + str(SNR3))


#R-Peak Detection
fs = sampling_freq
detectors = Detectors(fs)

x1 = fsignals.T
x1 = x1.flatten()
x2 = y.T
x2 = x2.flatten()
x3 = signalsW.T
x3 = x3.flatten()
x4 = signals0.T
x4 = x4.flatten()

#Wavelet Rpeaks
x1 = r_isolate_wavelet(x1,sampling_freq,sampTo)
x2 = r_isolate_wavelet(x2,sampling_freq,sampTo)
x3 = r_isolate_wavelet(x3,sampling_freq,sampTo)
x4 = r_isolate_wavelet(x4,sampling_freq,sampTo)

peaks1, _ = signal.find_peaks(x1, prominence=1, distance=200)
np.diff(peaks1)
peaks2, _ = signal.find_peaks(x2, prominence=1, distance=200)
np.diff(peaks2)
peaks3, _ = signal.find_peaks(x3, prominence=1, distance=200)
np.diff(peaks3)
peaks4, _ = signal.find_peaks(x4, prominence=1, distance=200)
np.diff(peaks4)

plt.plot(x1)
plt.plot(peaks1, x1[peaks1], "x")
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Butter Filtered w/ Peaks')
plt.savefig('Plots/Digital RPeak 300.png')
plt.show()

plt.plot(x2)
plt.plot(peaks2, x2[peaks2], "x")
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('RC Filtered w/ Peaks')
plt.savefig('Plots/Analog RPeak 300.png')
plt.show()

plt.plot(x3)
plt.plot(peaks3, x3[peaks3], "x")
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Wavelet w/ Peaks')
plt.savefig('Plots/Wavelet RPeak 300.png')
plt.show()

plt.plot(x4 + 3)
plt.plot(x3 + 2)
plt.plot(x2 + 1)
plt.plot(x1)
plt.plot(peaks4, x4[peaks4] + 3, "kx")
plt.plot(peaks3, x3[peaks3] + 2, "kx")
plt.plot(peaks2, x2[peaks2] + 1, "kx")
plt.plot(peaks1, x1[peaks1], "kx")
plt.grid(False)
plt.legend(['True ECG', 'Wavelet Filtered ECG', 'Butter Filtered ECG', 'RC Filtered ECG'])
plt.xlabel('Time (Sec)')
plt.title('Peaks Analog & Digital')
plt.savefig('Plots/Peaks Analog & Digital.png')
plt.show()


r_peaks1 = detectors.engzee_detector(x1)
r_peaks2 = detectors.engzee_detector(x2)
r_peaks3 = detectors.engzee_detector(x3)
r_peaks4 = detectors.engzee_detector(x4)

plt.plot(x4 + 3)
plt.plot(x3 + 2)
plt.plot(x2 + 1)
plt.plot(x1)
plt.plot(r_peaks4, x4[r_peaks4] + 3, "kx")
plt.plot(r_peaks3, x3[r_peaks3] + 2, "kx")
plt.plot(r_peaks2, x2[r_peaks2] + 1, "kx")
plt.plot(r_peaks1, x1[r_peaks1], "kx")
plt.grid(False)
plt.legend(['True ECG', 'Butter Filtered ECG', 'RC Filtered ECG'])
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Engzee Analog & Digital')
plt.savefig('Plots/Engzee Analog & Digital.png')
plt.show()


ms1 = np.mean(signals0**2)
ms2 = np.mean((signals-signals0)**2)
SNR4 = 10 * np.log(ms1/ms2) #one method of SNR calculation in decibels
print("Real SNR: " + str(SNR4))


#from DWT_denoising import DWT_denoising -> for wavelet


'''
Noisy - the noisy database signal -> signals
Clean - the clean database signal -> signals0
filtered - the filtered signal -> fsignals
'''