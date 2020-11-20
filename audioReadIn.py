import pyaudio
import wave
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fft
from DWT_ECG import DWT_denoise, r_isolate_wavelet
from biosppy.signals import ecg
import heartpy
import hrvanalysis
import pyhrv
from HRV_manual import get_hrv

'''
CHUNK = 528
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = np.array([])

b, a = signal.butter(6, 40/RATE, 'low')  # point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”)

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    processed = signal.filtfilt(b, a, np.frombuffer(data, np.int16), axis=0)
    frames = np.append(frames, processed)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()
time = np.linspace(0, RECORD_SECONDS, RATE * RECORD_SECONDS, endpoint=False)
print(frames.shape)
plt.plot(frames)
plt.show()
'''

RECORD_SECONDS = (5*60) + 1

CHUNK = 1024
FORMAT = pyaudio.paInt16 #can do float32 (probably want to)
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "Modified.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

#Read in signals
RATE, signals = wavfile.read("Modified.wav")

# Resample data
length=len(signals)
new_len=round(length/2)
new_rate = 500
sampTo = (new_rate*RECORD_SECONDS)
number_of_samples = round(len(signals) * float(new_rate) / RATE)
signals = signal.resample(signals, number_of_samples)

#Filter data
f0 = 60.0  # Frequency to be removed from signal (Hz)
Q = 30.0  # Quality factor
b, a = signal.iirnotch(f0, Q, new_rate)
processed = signal.lfilter(b,a,signals,axis=0)
b, a = signal.butter(2, 5/new_rate, 'low')
processed = signal.lfilter(b, a, processed, axis=0)
b, a = signal.iirnotch(f0, Q, new_rate)
processed = signal.lfilter(b,a,processed,axis=0)

#get rid of 1st second and append processed
processed = processed[new_rate:sampTo]

peaks, _ = signal.find_peaks(processed, prominence=max(processed[0:1000]), distance=new_rate/2)
np.diff(peaks)

'''
processed = processed[peaks[0]:peaks[-1]]
processed = np.append(processed, processed)
processed = np.append(processed, processed)
processed = np.append(processed, processed)

peaks, _ = signal.find_peaks(processed, prominence=max(processed[0:new_rate]), distance=new_rate/3) #try 1.5
np.diff(peaks)
'''

wdata,measures = get_hrv(peaks,processed[peaks],new_rate,'fft')

print('lf/hf ratio = %.3f' %measures['lf/hf'])
print ("Average Heart Beat is: %.01f" %measures['bpm'])

'''
# heartpy hrv analysis
rr = heartpy.analysis.calc_rr(peaks, new_rate)
rr['RR_list_cor']=rr['RR_list']
wd, m = heartpy.analysis.calc_fd_measures(method = 'fft', working_data = rr)
wd['peaklist']= processed
print('%.3f' %m['lf/hf'])

something = print(hrvanalysis.get_frequency_domain_features(peaks))
something2 = pyhrv.frequency_domain.welch_psd(peaks)
something3 = pyhrv.frequency_domain.ar_psd(peaks)
something4 = pyhrv.frequency_domain.lomb_psd(peaks)
'''

# plot shit
plt.plot(processed)
plt.plot(peaks, processed[peaks], "rx")
plt.show()

print('something')

#plt.plot(wd['RR_list'])
#plt.show()