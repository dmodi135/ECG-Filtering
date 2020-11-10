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
WAVE_OUTPUT_FILENAME = "output.wav"

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

RATE, signals = wavfile.read("output.wav")

length=len(signals)
new_len=round(length/2)
new_rate = 500
sampTo = (new_rate*RECORD_SECONDS)

# Resample data
number_of_samples = round(len(signals) * float(new_rate) / RATE)
signals = signal.resample(signals, number_of_samples)

b, a = signal.butter(2, 5/new_rate, 'low')
signals = signal.lfilter(b, a, signals, axis=0)
f0 = 60.0  # Frequency to be removed from signal (Hz)
Q = 30.0  # Quality factor
# Design notch filter
b, a = signal.iirnotch(f0, Q, new_rate)
processed = signal.lfilter(b,a,signals,axis=0)

processed = processed[new_rate:sampTo]

#x1 = r_isolate_wavelet(processed.flatten(),new_rate,len(processed.flatten()))
peaks, _ = signal.find_peaks(processed, prominence=max(processed[0:1000]), distance=new_rate/2)
np.diff(peaks)

rr = heartpy.analysis.calc_rr(peaks, new_rate)
rr['RR_list_cor']=rr['RR_list']
wd, m = heartpy.analysis.calc_fd_measures(method = 'fft', working_data = rr)
wd['peaklist']= processed
print('%.3f' %m['lf/hf'])
print('something')

#fig=plt.figure()
#ax1=fig.add_subplot(1,2,1)
plt.plot(processed[0:sampTo])
plt.plot(peaks, processed[peaks], "rx")
plt.show()

plt.plot(wd['RR_list'])
plt.show()
'''
arr2 = read("MiaMinute.wav")
signals2 = np.array(arr2[1])
#signals = signals[0:sampTo]
#out = ecg.ecg(signal=signals, sampling_rate=44100., show=True)

b2, a2 = signal.butter(2, 5/RATE, 'low')
processed2 = signal.lfilter(b2, a2, signals2, axis=0)
#processed = DWT_denoise(signals, fs, sampTo)
processed2 = processed2[0:sampTo]

#x1 = r_isolate_wavelet(processed.flatten(),fs,sampTo)
peaks2, _ = signal.find_peaks(processed2, prominence=100, distance=20000)
np.diff(peaks2)

ax2=fig.add_subplot(1,2,2)
plt.plot(processed2[0:sampTo])
plt.plot(peaks2, processed2[peaks2], "rx")
plt.show()
'''
