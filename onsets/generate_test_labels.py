import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *

from onsets.panotti.models import *
from onsets.panotti.datautils import *
from predict_class import predict_one


model = load_model('weights.hdf5')

def test():
    class_names = get_class_names('Preproc/Test/')

    window_size = .1
    sr = 44100

    plt.figure(figsize=(12, 8))
    files =  tqdm([f for f in os.listdir('raw_data') if f.endswith('.wav')])
    for f in files:
        files.set_description("Processing %s" % f)
        path = (os.path.join('raw_data', f))
        increment = int(sr * window_size)
        y, sr  = librosa.load(path, sr)
        outfile = open(os.path.join('raw_data/generated/', f.replace('.wav', '.txt')), 'w')
        for i in range(sr, len(y), increment):
            start = i/sr
            end = (i + increment)/sr
            half = increment/sr
            clip = librosa.util.fix_length(y[i:i+increment], increment)
            outpath = 'Samples/no/%s-%d-%d.wav'%(f.replace(' ','-'), i, i+increment)
            
            if class_names[np.argmax(predict_one(clip, sr, model))] == 'yes':
                ons = librosa.onset.onset_detect(y=clip, sr=sr, units='samples')/sr
                for o in [ons[0]]:
                    outfile.write('\t'.join([str(start+o), str(start+o), '\n']))
        
        outfile.close()
              

if __name__ == '__main__':
    test()