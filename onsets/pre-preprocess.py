import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
import random

from onsets.constants import SR, ONSET_WINDOW_TIME

ONLY_YES_FILES = [
    '4-10-Audio.wav',
    '0006-10-Audio.wav'
]

def process_rawdata():

    window_size = ONSET_WINDOW_TIME
    random_offset = .025
    sr = SR

    files =  tqdm([f for f in os.listdir('raw_data') if f.endswith('.wav')])
    for f in files:
        files.set_description("Processing %s" % f)
        path = (os.path.join('raw_data', f))
        label = path.replace('audio', 'labels').replace('.wav', '.txt')
        label_lines =  open(label, 'r').readlines()
        label_times = np.array([np.float(s.split('\t')[0]) for s in label_lines])
        label_names = [s.split('\t')[-1] for s in label_lines]

        y, sr  = librosa.load(path, sr)
        increment = int(sr * window_size)
        for i in range(0, len(y), increment):
            start = i/sr
            end = (i + increment)/sr
            outpath = 'Samples/no/%s-%02f-%02f.wav'%(f.replace(' ','-'), float(i)/sr, (float(i)+increment)/sr)
            clip = librosa.util.fix_length(y[i:i+increment], increment)
            if np.any(label_times[(label_times >= start) & (label_times <= end)]):
                label_start = label_times[0]
                label_name = label_names[0]
                
                # remove 1st label from the list
                label_names = label_names[1:]
                label_times = label_times[1:]

                if 'bass' in label_name:
                    continue
                if 'singing' in label_name:
                    continue
                    
                start_sample = int((label_start - np.random.normal(window_size/2.0, random_offset))*sr)  
                clip = y[start_sample:start_sample+increment]
                outpath = 'Samples/yes/%s-%02f-%02f.wav'%(f.replace(' ','-'), start_sample/sr, (start_sample+increment)/sr)
                
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            if f not in ONLY_YES_FILES or '/yes/' in outpath:
                librosa.output.write_wav(outpath, clip, sr)

if __name__ == '__main__':
    process_rawdata()