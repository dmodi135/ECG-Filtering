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

plt.rcParams.update({'font.size':36})
os.chdir("/Users/dhruvmodi/Desktop/ECG-Filtering/mit-database")

sampTime = 10
sampling_freq = 360
sampTo = sampTime*sampling_freq

#read in noisy data for 2 seconds
noisy, fields = wfdb.rdsamp('118e06', channels=[0], sampfrom=108000, sampto=108000 + sampTo) #194400)
#read in clean data for 2 seconds
clean, fields0 = wfdb.rdsamp('118', channels=[0], sampfrom=108000, sampto=108000 + sampTo) #194400)
#assign variables and create time-frame
sampling_duration = sampTime
number_of_samples = fields.get('sig_len')
time = np.linspace(0, sampling_duration, number_of_samples, endpoint=False)
os.chdir('/Users/dhruvmodi/Desktop/ECG-Filtering/')
clean = clean-np.mean(clean)
noisy = noisy-np.mean(noisy)

'''
b, a = signal.butter(2, 0.05, 'high')
clean = signal.lfilter(b, a, clean, axis=0)
b, a = signal.butter(2, 0.05, 'high')
noisy = signal.lfilter(b, a, noisy, axis=0)

difference1 = 0-signals[0]
signals = signals + difference1
difference2 = 0-signals0[0]
signals0 = signals0 + difference2
'''

#plot noisy data vs clean
OG_plot = plt.figure(figsize=(40,20))
plt.plot(time, clean, 'b-', label='signal')
plt.plot(time, noisy, 'g-', label='noisy signal')
plt.legend(['Clean ECG', 'Noisy ECG'])
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Unfiltered vs. True ECG')
plt.savefig('Plots/Original ECG vs Noisy ECG.png')
plt.show()
#plot noise data vs clean (it's different)
noise = noisy-clean #unfiltered - filtered should just leave noise
plt.plot(time, noise, 'k-')
plt.plot(time, clean, 'g-')
plt.legend(['Noisy ECG', 'Pure Noise'])
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Unfiltered vs. Noise')
plt.savefig('Plots/Noisy.png')
plt.show()


#do and plot filter
N  = 2 # Filter order
Wn = 0.08 # Cutoff frequency bw 0 & 1 (30 Hz cutoff bcz of fourier which is 500(fs) * 0.08)
b, a = signal.butter(N, Wn, 'low') #point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”)
butter_filtered = signal.lfilter(b, a, noisy, axis=0)
butter_plot = plt.figure(figsize=(40,20))
plt.plot(time, noisy)
plt.plot(time, clean, 'black')
plt.plot(time, butter_filtered,color='purple') #subtracting just for visuals
plt.legend(['Noisy ECG', 'Clean ECG', 'Butter Filtered ECG'])
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Butter Filtered Data')
plt.savefig('Plots/2nd Order Butter @ 30.png')
plt.show()
#potential SNR calculation
ms1 = np.mean(butter_filtered**2)
ms2 = np.mean((butter_filtered-clean)**2)
SNR = 10 * np.log(ms1/ms2) #one method of SNR calculation in decibels
print("2nd Order Butter SNR: " + str(SNR))


# ## Fourier Transform
# noisy = noisy.T
# f = fft(noisy)
# f = np.abs(f)
# freq = np.fft.fftfreq(sampTo, d=1/sampling_freq)
# plt.plot(freq, f.T)
# #plt.plot(np.linspace(0, 30,30), np.linspace(0, 30,30)) #check a good cutoff freq
# plt.legend(['Frequency Magnitudes'])
# plt.grid(False)
# plt.xlabel('Frequency')
# plt.ylabel('mV')
# plt.title('Fourier Transformed Data')
# plt.savefig('Plots/Fourier.png')
# plt.show()
# noisy = noisy.T


#transfer function; 2nd order RC filter using 1.33k resistors and 4u capacitors (look at bode plots)
#http://sim.okawa-denshi.jp/en/CRCRtool.php

sys = signal.lti([35332.69263384],[1,563.90977443609,35332.69263384]) #cutoff freq of 30Hz
#sys = signal.TransferFunction([35332.69263384],[1,563.90977443609,35332.69263384]) #cutoff freq 30Hz

#Transfer function does same -> function is 35332.69263384 / s^2 + 563.90977443609s + 35332.69263384
#^ created from 2 Resistors at 1.33 kilo Ohms and 2 Caps at 4 micro Farrads

t = np.linspace(0, sampTime, sampTo)
tout, rc_filtered, x = signal.lsim2(sys, noisy, t)

rc_plot=plt.figure(figsize=(40,20))
plt.plot(t, noisy)
plt.plot(time, clean, 'black')
plt.plot(t, rc_filtered,color='red')
plt.legend(['Unfiltered ECG', 'Clean ECG', 'RC Filtered ECG'])
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('RC Filtered Data')
plt.savefig('Plots/2nd Order RC @ 30.png')
plt.show()
#potential SNR calculation
ms1 = np.mean(rc_filtered**2)
ms2 = np.mean((rc_filtered-clean)**2)
SNR2 = 10 * np.log(ms1/ms2) #one method of SNR calculation in decibels
print("2nd Order RC SNR: " + str(SNR2))


#Wavelet Filter
noisy_flat= noisy.flatten()
wavelet_filtered = DWT_denoise(noisy_flat, sampling_freq, sampTo)
wavelet_plot=plt.figure(figsize=(40,20))
plt.plot(t, noisy)
plt.plot(time, clean, 'black')
plt.plot(t, wavelet_filtered,color='orange')
plt.legend(['Unfiltered ECG', 'Clean ECG', 'Wavelet Filtered ECG'])
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Wavelet Filtered Data')
plt.savefig('Plots/Wavelet Filtered.png')
plt.show()
#potential SNR calculation
ms1 = np.mean(wavelet_filtered**2)
ms2 = np.mean((wavelet_filtered-clean)**2)
SNR3 = 10 * np.log(ms1/ms2) #one method of SNR calculation in decibels
print("Wavelet SNR: " + str(SNR3))


#R-Peak Detection
fs = sampling_freq
detectors = Detectors(fs)

xbutter = butter_filtered.T
xbutter = xbutter.flatten()
xrc = rc_filtered.T
xrc = xrc.flatten()
xwavelet = wavelet_filtered.T
xwavelet = xwavelet.flatten()
xclean = clean.T
xclean = xclean.flatten()
xnoisy= noisy.T
xnoisy=xnoisy.flatten()

peaks_butter, _ = signal.find_peaks(xbutter, prominence=1, distance=200)
np.diff(peaks_butter)
peaks_rc, _ = signal.find_peaks(xrc, prominence=1, distance=200)
np.diff(peaks_rc)
peaks_wavelet, _ = signal.find_peaks(xwavelet, prominence=1, distance=200)
np.diff(peaks_wavelet)
peaks_clean, _ = signal.find_peaks(xclean, prominence=1, distance=200)
np.diff(peaks_clean)
peaks_noisy, _ = signal.find_peaks(xnoisy, prominence=1, distance=200)
np.diff(peaks_noisy)


#Plotting regular r-peak detection w findpeaks

'''
plt.plot(xbutter)
plt.plot(peaks_butter, xbutter[peaks_butter], "x")
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Butter Filtered w/ Peaks')
plt.savefig('Plots/Digital RPeak 300.png')
plt.show()

plt.plot(xrc)
plt.plot(peaks_rc, xrc[peaks_rc], "x")
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('RC Filtered w/ Peaks')
plt.savefig('Plots/Analog RPeak 300.png')
plt.show()

plt.plot(xwavelet)
plt.plot(peaks_wavelet, xwavelet[peaks_wavelet], "x")
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Wavelet w/ Peaks')
plt.savefig('Plots/Wavelet RPeak 300.png')
plt.show()
'''

filters=plt.figure(figsize=(40,20))
plt.grid(False)
#plt.legend(['True ECG', 'Wavelet Filtered ECG', 'Butter Filtered ECG', 'RC Filtered ECG'])
plt.xlabel('Time (Sec)')
plt.title('Find_Peaks Analog & Digital',y=1.08)
plt.axis('off')

ax_clean=filters.add_subplot(5,1,1)
ax_clean.title.set_text("Clean Signal")
plt.plot(xclean)
plt.plot(peaks_clean, xclean[peaks_clean], "ko")

ax_noisy=filters.add_subplot(5,1,2)
ax_noisy.title.set_text("Noisy Signal")
plt.plot(xnoisy,color='purple')
plt.plot(peaks_noisy,xnoisy[peaks_noisy],"ko")

ax_wavelet=filters.add_subplot(5,1,3)
ax_wavelet.title.set_text("Wavelet Filtered")
plt.plot(xwavelet,color='orange')
plt.plot(peaks_wavelet, xwavelet[peaks_wavelet], "ko")

ax_rc=filters.add_subplot(5,1,4)
ax_rc.title.set_text("RC Filtered")
plt.plot(xrc,color='red')
plt.plot(peaks_rc, xrc[peaks_rc], "ko")

ax_butter=filters.add_subplot(5,1,5)
ax_butter.title.set_text("Butter Filtered")
plt.plot(xbutter,color='green')
plt.plot(peaks_butter, xbutter[peaks_butter], "ko")

plt.tight_layout()
plt.savefig('Plots/Peaks Analog & Digital.png')
plt.show()

#r peak detection with engzee
r_peaks_butter = detectors.engzee_detector(xbutter)
r_peaks_rc = detectors.engzee_detector(xrc)
r_peaks_wavelet = detectors.engzee_detector(xwavelet)
r_peaks_clean = detectors.engzee_detector(xclean)
r_peaks_noisy=detectors.engzee_detector(xnoisy)


rpeaks=plt.figure(figsize=(40,20))
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Engzee Analog & Digital',y=1.08)
plt.axis('off')

ax_clean=rpeaks.add_subplot(5,1,1)
ax_clean.title.set_text("Clean Peaks")
plt.plot(xclean)
plt.plot(r_peaks_clean, xclean[r_peaks_clean], "ko")

ax_noisy=rpeaks.add_subplot(5,1,2)
ax_noisy.title.set_text("Noisy Signal")
plt.plot(xnoisy,color='purple')
plt.plot(peaks_noisy,xnoisy[peaks_noisy],"ko")

ax_wavelet=rpeaks.add_subplot(5,1,3)
ax_wavelet.title.set_text("Wavelet Peaks")
plt.plot(xwavelet,color='orange')
plt.plot(r_peaks_wavelet, xwavelet[r_peaks_wavelet], "ko")

ax_rc=rpeaks.add_subplot(5,1,4)
ax_rc.title.set_text("RC Peaks")
plt.plot(xrc,color='red')
plt.plot(r_peaks_rc, xrc[r_peaks_rc], "ko")

ax_butter=rpeaks.add_subplot(5,1,5)
ax_butter.title.set_text("Butter Peaks")
plt.plot(xbutter,color='green')
plt.plot(r_peaks_butter, xbutter[r_peaks_butter], "ko")

plt.tight_layout()
plt.savefig('Plots/Engzee Analog & Digital.png')
plt.show()


#Wavelet Rpeaks
xwbutter = r_isolate_wavelet(xbutter,sampling_freq,sampTo)
xwrc = r_isolate_wavelet(xrc,sampling_freq,sampTo)
xwwavelet = r_isolate_wavelet(xwavelet,sampling_freq,sampTo)
xwclean = r_isolate_wavelet(xclean,sampling_freq,sampTo)
xwnoisy= r_isolate_wavelet(xnoisy,sampling_freq,sampTo)


r_peaks_butter_wavelet = detectors.engzee_detector(xwbutter)
r_peaks_rc_wavelet = detectors.engzee_detector(xwrc)
r_peaks_wavelet_wavelet = detectors.engzee_detector(xwwavelet)
r_peaks_clean = detectors.engzee_detector(xclean)
r_peaks_noisy_wavelet=detectors.engzee_detector(xwnoisy)

#plotting partial wavelet deconstruction r peak detection with engzee
wave_rpeaks=plt.figure(figsize=(40,20))
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Partial Wavelet Reconstruction R-Peaks + Engzee',y=1.08)
plt.axis('off')

ax_clean=wave_rpeaks.add_subplot(5,1,1)
ax_clean.title.set_text("Clean Peaks")
plt.plot(xclean)
plt.plot(r_peaks_clean, xclean[r_peaks_clean], "ko")

ax_noisy=wave_rpeaks.add_subplot(5,1,2)
ax_noisy.title.set_text("Noisy Signal")
plt.plot(xnoisy,color='purple')
plt.plot(r_peaks_noisy_wavelet,xnoisy[r_peaks_noisy_wavelet],"ko")

ax_wavelet=wave_rpeaks.add_subplot(5,1,3)
ax_wavelet.title.set_text("Wavelet Peaks")
plt.plot(xwavelet,color='orange')
plt.plot(r_peaks_wavelet_wavelet, xwavelet[r_peaks_wavelet_wavelet], "ko")

ax_rc=wave_rpeaks.add_subplot(5,1,4)
ax_rc.title.set_text("RC Peaks")
plt.plot(xrc,color='red')
plt.plot(r_peaks_rc_wavelet, xrc[r_peaks_rc_wavelet], "ko")

ax_butter=wave_rpeaks.add_subplot(5,1,5)
ax_butter.title.set_text("Butter Peaks")
plt.plot(xbutter,color='green')
plt.plot(r_peaks_butter_wavelet, xbutter[r_peaks_butter_wavelet], "ko")

plt.tight_layout()
plt.savefig('Plots/PartialWaveletReconstructionEngzee.png')
plt.show()

#partial wavelet deconstruction + find_peaks
wpeaks_butter, _ = signal.find_peaks(xwbutter, prominence=1, distance=200)
np.diff(peaks_butter)
wpeaks_rc, _ = signal.find_peaks(xwrc, prominence=1, distance=200)
np.diff(peaks_rc)
wpeaks_wavelet, _ = signal.find_peaks(xwwavelet, prominence=1, distance=200)
np.diff(peaks_wavelet)
wpeaks_clean, _ = signal.find_peaks(xwclean, prominence=1, distance=200)
np.diff(wpeaks_clean)
wpeaks_noisy, _ = signal.find_peaks(xwnoisy, prominence=1, distance=200)
np.diff(wpeaks_noisy)

#plotting r peak detection with partial wavelet deconstruction and find peaks
w_rpeaks=plt.figure(figsize=(40,20))
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Partial Wavelet Reconstruction R-Peaks + find_peaks',y=1.08)
plt.axis('off')

ax_clean=w_rpeaks.add_subplot(5,1,1)
ax_clean.title.set_text("Clean Peaks")
plt.plot(xclean)
plt.plot(peaks_clean, xclean[peaks_clean], "ko")

ax_noisy=w_rpeaks.add_subplot(5,1,2)
ax_noisy.title.set_text("Noisy Signal")
plt.plot(xnoisy,color='purple')
plt.plot(wpeaks_noisy,xnoisy[wpeaks_noisy],"ko")

ax_wavelet=w_rpeaks.add_subplot(5,1,3)
ax_wavelet.title.set_text("Wavelet Peaks")
plt.plot(xwavelet,color='orange')
plt.plot(wpeaks_wavelet, xwavelet[wpeaks_wavelet], "ko")

ax_rc=w_rpeaks.add_subplot(5,1,4)
ax_rc.title.set_text("RC Peaks")
plt.plot(xrc, color='red')
plt.plot(wpeaks_rc, xrc[wpeaks_rc], "ko")

ax_butter=w_rpeaks.add_subplot(5,1,5)
ax_butter.title.set_text("Butter Peaks")
plt.plot(xbutter, color='green')
plt.plot(wpeaks_butter, xbutter[wpeaks_butter], "ko")

plt.tight_layout()
plt.savefig('Plots/PartialWaveletReconstructionFind_Peaks.png')
plt.show()

ms1 = np.mean(clean**2)
ms2 = np.mean(noise**2)
SNR4 = 10 * np.log(ms1/ms2) #one method of SNR calculation in decibels
print("Real SNR: " + str(SNR4))


#from DWT_denoising import DWT_denoising -> for wavelet


'''
Noisy - the noisy database signal -> signals
Clean - the clean database signal -> signals0
filtered - the filtered signal -> fsignals
'''

exit(0)
