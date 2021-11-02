import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
import os
import glob

csv_paths = []
csv_paths = glob.glob("../data/Part_4/*.csv")


for i in range(len(csv_paths)):
    # Reading from CSV
    Y = np.genfromtxt(csv_paths[i], delimiter=',')
    O1P = Y[0, :1000] # PPG
    BP  = Y[1, :1000] # ABP
    O1E = Y[2, :1000] # ECG
    Fs  = 125 # Sampling Frequency
    Ts  = 1/Fs # Time Period
    T   = np.arange(0, 8, 0.008)
    W1  = 0.5*2/Fs # Lowpass Frequency
    W2  = 5*2/Fs  # HighPass Frequency
    [b,a] = signal.butter(3, [W1, W2], btype='bandpass') # Bandpass Digital Filter
    FP  = signal.filtfilt(b, a, O1P) # Filtering with the filter designed
    Fy  = np.gradient(FP)  # Taking gradient

    # Removing the negative gradients
    for j in range(1000):
        if Fy[j] <= 0:
            Fy[j] = 0
    
    # Moving sum
    window = np.ones(3, dtype=int)
    T1  = np.convolve(Fy, window, "same")

    W1  = 0.5*2/Fs # LowPass Frequency
    W2  = 40*2/Fs # HighPass Frequency
    [b,a] = signal.butter(3, [W1, W2], btype='bandpass') # Design bandpass filter
    FP1 = signal.filtfilt(b, a, O1E) # Filtering with the filter designed
    A   = signal.detrend(FP1) # removes linear trend from dataset
    E   = signal.detrend(FP)  # removes linear trend from dataset

    # Moving Max
    D   = np.max(sliding_window_view(T1, window_shape = 3), axis = 1)

    #Find local maximas
    [pk1, _] = signal.find_peaks(D)
    h   = np.zeros((1000)) # Initiating zero array

    # marking maximas
    for i in pk1:
        h[i] = 1

    h[h==1] = D[pk1] # Saving peaks to h

    # Finding correlation between
    # detrended ECG and peaks of ppg
    C = signal.correlate(A, h, mode="full")
    Lag = signal.correlation_lags(len(A), len(h), mode="full")

    # Finding the index with max Correlation
    I = np.argmax(abs(C))

    # Finding difference which is equal to PTT
    Diff = Lag[I] / Fs

    # Writing diff to csv
    with open("../FinalDataset/ptt.csv", "a") as f:
        f.write(f"{abs(Diff)}\n")
