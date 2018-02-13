import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *

from panotti import models as sound_class
from panotti.datautils import *
from predict_class import predict_one

from onsets.predict_class import predict_one as predict_onset
from onsets.panotti import models as onsets

onset_model = onsets.load_model('onsets/weights.hdf5')
sound_model = sound_class.load_model('weights.hdf5')

def test():
    class_names = get_class_names('Preproc/Test/')

    plt.figure(figsize=(12, 8))
    files =  tqdm([f for f in os.listdir('raw_data') if f.endswith('.wav')])
    os.makedirs('raw_data/generated/', exist_ok=True)
    for f in files:
        files.set_description("Processing %s" % f)
        path = (os.path.join('raw_data', f))
        increment = int(44100 * .15)
        y, sr  = librosa.load(path, 44100)
        outfile = open(os.path.join('raw_data/generated/', f.replace('.wav', '.txt')), 'w')
        for i in range(0, len(y), increment):
            start = i/44100
            end = (i + increment)/44100
            half = increment/44100
            clip = librosa.util.fix_length(y[i:i+increment], increment)

            print (predict_onset(clip, sr, onset_model))
            
            label = class_names[np.argmax(predict_one(clip, sr, model))]
            if not label == 'silence':
                ons = librosa.onset.onset_detect(y=clip, sr=44100, units='samples')/44100.
                outfile.write('\t'.join([str(start), str(end), label, '\n']))
        
        outfile.close()
              

if __name__ == '__main__':
    test()