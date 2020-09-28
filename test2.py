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

if "record" is not locals():
    os.chdir("/Users/dhruvmodi/Desktop/ECG-Filtering/ecg-database/Person_01/")

import wfdb

# Demo 1 - Read a WFDB record using the 'rdrecord' function into a wfdb.Record object.
# Plot the signals, and show the data.
#this reads and plots a noisy record from the MIT stress test database, along with all the information about the record/patient
#NOTE: do not use this function if you want to further manipulate the ECG data. instead, use wfdb.rdsamp
record = wfdb.rdrecord('rec_1')
wfdb.plot_wfdb(record=record, title='MIT record 1')
display(record.__dict__)

#%%

# Demo 2 - Read certain channels and sections of the WFDB record using the simplified 'rdsamp' function
# which returns a numpy array and a dictionary. Show the data.®
#this section reads a portion of the same record as above (from sample 100 to sample 15000)
#returns a "signals" (a numpy array) and "fields"(patient and signal info)
#note: this ecg has data from 2 channels (2 different leads), if you want data from a specific lead, you have to
#select the correct channel that corresponds to the leads(channels start at 0, not 1)

#NOTE: use this function whenever you want to work with the ECG data (not rdrecord)

signals, fields = wfdb.rdsamp('rec_1', channels=[0], sampfrom=0, sampto= 1000) #194400)
display(signals)
display(fields)

signals7, fields7 = wfdb.rdsamp('rec_1', channels=[1], sampfrom=0, sampto= 1000) #194400)
display(signals7)
display(fields7)

z = fft(signals)
display(np.max(z))
#signals = np.abs(signals)

sampling_freq = 500
sampling_duration = 2
number_of_samples = int(sampling_freq * sampling_duration)
time = np.linspace(0, sampling_duration, number_of_samples, endpoint=False)

plt.plot(time, signals, 'b-', label='signal')
plt.plot(time, signals7, 'g-', linewidth=2, label='filtered signal')
plt.show()

N  = 3    # Filter order
Wn = .1 #1-0.735 # Cutoff frequency
b, a = signal.butter(N, Wn, 'low')
fsignals = signal.filtfilt(b, a, signals, axis=0)

plt.plot(time, signals, 'b-', label='signal')
plt.plot(time, fsignals-0.05, 'g-', linewidth=2, label='filtered signal') #subtracting just for visuals
plt.show()

#signals = signals / 1000
#fsignals = fsignals / 1000
#signal7 = signals7 / 1000

noise = signals-fsignals #unfiltered - filtered should just leave noise

'''
rms1 = np.sqrt(np.mean(fsignals**2))
rms2 = np.sqrt(np.mean(noise**2))
SNR = (rms1/rms2)**2
'''

ms1 = np.mean(fsignals**2)
ms2 = np.mean(noise**2)
SNR = 10 * np.log(ms1/ms2)

print(SNR)


'''
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def run():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 500 #5000.0
    lowcut = 10 #500.0
    highcut = 230 #1250.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)

    # Filter a noisy signal.
    T = 2
    t = np.linspace(0, T, 1000, endpoint=False)
    x = signals
    plt.figure(2)
    plt.clf()
    plt.plot(t, x)

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y)
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')

    plt.show()

    return y

y = run()
'''

'''
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

signals = np.abs(signals)
# Filter requirements.
order = 2
fs = 500      # sample rate, Hz
cutoff = 100 # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)


# Demonstrate the use of the filter.
# First make some data to be filtered.
T = 2         # seconds
n = int(T * fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)
# "Noisy" data.  We want to recover the 1.2 Hz signal from this.
#data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(signals, cutoff, fs, order)

plt.plot(t, signals, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()
'''

'''
# Filter requirements.
T = 2.0         # Sample Period
fs = 500.0       # sample rate, Hz
cutoff = 200     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hznyq = 0.5 * fs  # Nyquist Frequencyorder = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)

nyq = 0.5 * fs  # Nyquist Frequency
order = 6      # sin wave can be approx represented as quadratic

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, padlen=0, method='gust')
    return y

y = butter_lowpass_filter(signals, cutoff, fs, order)
plt.plot(t, signals, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.show()
'''

'''
fig = go.Figure()
fig.add_trace(go.Scatter(
            y = data,
            line =  dict(shape =  'spline' ),
            name = 'signal with noise'
            ))
fig.add_trace(go.Scatter(
            y = y,
            line =  dict(shape =  'spline' ),
            name = 'filtered signal'
            ))
fig.show()
'''

'''
order = 5
sampling_freq = 500
cutoff_freq = 200
sampling_duration = 2
number_of_samples = int(sampling_freq * sampling_duration)
time = np.linspace(0, sampling_duration, number_of_samples, endpoint=False)

normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
numerator_coeffs, denominator_coeffs = signal.butter(order,normalized_cutoff_freq)
filtered_signal = signal.lfilter(numerator_coeffs, denominator_coeffs, signals)

plt.plot(time, signals, 'b-', label='signal')
plt.plot(time, filtered_signal, 'g-', linewidth=2, label='filtered signal')
plt.legend()
plt.show()
'''


'''
order = 5
sampling_freq = 500
cutoff_freq = 200
sampling_duration = 2
number_of_samples = int(sampling_freq * sampling_duration)
time = np.linspace(0, sampling_duration, number_of_samples, endpoint=False)

def bandpass_filter(data, low, high, fs, order):
    """
    Does a bandpass filter over the given data.

    :param data: The data (numpy array) to be filtered.
    :param low: The low cutoff in Hz.
    :param high: The high cutoff in Hz.
    :param fs: The sample rate (in Hz) of the data.
    :param order: The order of the filter. The higher the order, the tighter the roll-off.
    :returns: Filtered data (numpy array).
    """

    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data)
    return y

y = bandpass_filter(signals, 5, 200, 500, 4)

plt.plot(time, signals, 'b-', label='signal')
plt.plot(time, y, 'g-', linewidth=2, label='filtered signal')
plt.legend()
plt.show()
'''
