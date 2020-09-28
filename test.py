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
    os.chdir("/Users/dhruvmodi/Desktop/ECG-Filtering/mit-database/")

import wfdb

# Demo 1 - Read a WFDB record using the 'rdrecord' function into a wfdb.Record object.
# Plot the signals, and show the data.
#this reads and plots a noisy record from the MIT stress test database, along with all the information about the record/patient
#NOTE: do not use this function if you want to further manipulate the ECG data. instead, use wfdb.rdsamp
record = wfdb.rdrecord('ma')
wfdb.plot_wfdb(record=record, title='MIT record 1')
display(record.__dict__)

#%%

# Demo 2 - Read certain channels and sections of the WFDB record using the simplified 'rdsamp' function
# which returns a numpy array and a dictionary. Show the data.
#this section reads a portion of the same record as above (from sample 100 to sample 15000)
#returns a "signals" (a numpy array) and "fields"(patient and signal info)
#note: this ecg has data from 2 channels (2 different leads), if you want data from a specific lead, you have to
#select the correct channel that corresponds to the leads(channels start at 0, not 1)

#NOTE: use this function whenever you want to work with the ECG data (not rdrecord)

signals, fields = wfdb.rdsamp('118e24', channels=[0], sampfrom=108600, sampto= 108600+360) #194400)
display(signals)
display(fields)

#signals7, fields7 = wfdb.rdsamp('118e12', channels=[1], sampfrom=0, sampto= 360) #194400)
#display(signals7)
#display(fields7)

time = np.linspace(0, 1, 360, endpoint=False)

z = fft(signals)
display(np.max(z))
#signals = np.abs(signals)

sampling_freq = 360
sampling_duration = 1
number_of_samples = int(sampling_freq * sampling_duration)
time = np.linspace(0, sampling_duration, number_of_samples, endpoint=False)

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


