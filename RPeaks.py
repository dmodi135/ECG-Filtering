import numpy as np
import matplotlib.pyplot as plt
import pathlib
from ecgdetectors import Detectors
import os
import wfdb

#current_dir = pathlib.Path('example_data').resolve()

#example_dir = current_dir.parent/'example_data'/'ECG.tsv'

os.chdir("/Users/dhruvmodi/Desktop/ECG-Filtering/ecg-database/Person_01/")
signals, fields = wfdb.rdsamp('rec_1', channels=[0], sampfrom=0, sampto= 1000)

#unfiltered_ecg_dat = np.loadtxt(example_dir)
unfiltered_ecg = signals
fs = 500

detectors = Detectors(fs)

#r_peaks = detectors.two_average_detector(unfiltered_ecg)
#r_peaks = detectors.matched_filter_detector(unfiltered_ecg,"templates/template_250hz.csv")
#r_peaks = detectors.swt_detector(unfiltered_ecg)
r_peaks = detectors.engzee_detector(unfiltered_ecg)
#r_peaks = detectors.christov_detector(unfiltered_ecg)
#r_peaks = detectors.hamilton_detector(unfiltered_ecg)
#r_peaks = detectors.pan_tompkins_detector(unfiltered_ecg)


plt.figure()
plt.plot(unfiltered_ecg)
plt.plot(r_peaks, unfiltered_ecg[r_peaks], 'ro')
plt.title('Detected R-peaks')

plt.show()