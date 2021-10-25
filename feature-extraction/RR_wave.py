import scipy as sp
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import math
import csv

csv_dir_path = "../data/Part_1"

csv_paths = []

for patNum in range(1,11):
    csv_paths.append(os.path.join(csv_dir_path, str(patNum)+".csv"))
    
csv_paths = []

for patNum in range(1,11):
    csv_paths.append(os.path.join(csv_dir_path, str(patNum)+".csv"))

for i in range(10):
    Y = np.genfromtxt(csv_paths[i], delimiter=",")
   
    # Calculating the number of readings in a file
    input_file = open(csv_paths[i],"r+")
    reader_file = csv.reader(input_file)
    lines = len(next(reader_file))
    #print(lines)
    ppg_Signal=[]
    BP_Signal=[]
    ECG_Signal=[]
    for i in range(0,lines,100):
        # Extracting row cooresponding to PPG signal because the first row of csv file is PPG signal
        if(i+100<lines):
            ppg_Signal = Y[0,i:i+100]
            # Extracting row cooresponding to BP signal
            BP_Signal  = Y[1,i:i+100]
            # Extracting row cooresponding to ECG signal
            ECG_Signal = Y[2,i:i+100]
            i+=100
        else:
            
            ppg_Signal = Y[0,i:]
            # Extracting row cooresponding to BP signal
            BP_Signal  = Y[1,i:]
            # Extracting row cooresponding to ECG signal
            ECG_Signal = Y[2,i:]
            i=i+100
        
        """if(len(ppg_Signal)!=100):
            print(len(ppg_Signal))
        """
        Fy  = np.gradient(ppg_Signal)

        # figure('PPG 1st derivative')
        # Uncomment below to check the graphs found from the first derivative of PPG
        """
        plt.plot(range(len(Fy)),Fy)
        plt.xlabel("PPG first derivative")
        plt.show()
        """

        Fy1 = np.gradient(Fy)

        # figure('PPG 2nd derivative')
        # Uncomment below to check the graphs found from the second derivative of PPG
        """
        plt.plot(range(len(Fy)),Fy)
        plt.xlabel("PPG second derivative")
        plt.show()
        """
        F = np.ones(100)
        np.append(F, ppg_Signal)
        np.append(F, BP_Signal)
        np.append(F, ECG_Signal)
        
        L = len(BP_Signal)

        # Sampling frequency = 125 Hz
        Fs = 125
        # Time vector based on sampling rate
        Ts = 1 / Fs

        T = np.arange(0, 8, 0.008) 
        
        #Find peaks or local maximum of PPG signals
        [peaks, location] = signal.find_peaks(ECG_Signal)
        
        print(peaks)
        
        # RR wave feature Extraction _______________________
        [RR_Wave_duration]=[]
        [s_no]=[]
        for i in range (len(peaks)-1):
            RR_Wave_duration.append(location[i+1]-location[i])
            s_no.append(i)
            print(s_no[i],' ',RR_Wave_duration[i])
            
            

        
        
    