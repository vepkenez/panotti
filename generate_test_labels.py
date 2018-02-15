import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *

from constants import AUDIO_WINDOW_SAMPLES, SR

from panotti import models as sound_class
from panotti.datautils import *
from predict_class import predict_one

from onsets.predict_class import predict_one as predict_onset
from onsets.panotti import models as onsets
from onsets.constants import ONSET_WINDOW_TIME, ONSET_WINDOW_SAMPLES

onset_model = onsets.load_model('onsets/weights.hdf5')

sound_model = sound_class.load_model('weights.hdf5')

def detect_onset(clip, sr):
    pred = predict_onset(clip, sr, onset_model)
    if ['no', 'yes'][np.argmax(pred)] == 'yes':
        ons = librosa.onset.onset_detect(y=clip, sr=sr, units='samples')
        if ons.any():
            return ons[-1]

def test():
    class_names = get_class_names('Preproc/Test/')
    print(class_names)

    plt.figure(figsize=(12, 8))
    files =  tqdm([f for f in os.listdir('raw_data') if f.endswith('.wav')])
    os.makedirs('raw_data/generated/', exist_ok=True)
    for f in files:
        files.set_description("Processing %s" % f)
        path = (os.path.join('raw_data', f))
        stride = int(SR * .02) # 20 milliseconds
        onset_increment = ONSET_WINDOW_SAMPLES
        sound_window = AUDIO_WINDOW_SAMPLES
        last_sound_sample = 0
        min_sound_spacing = .06 * SR # minimum resolution: 60 milliseconds =~ 1/32 note at 120bpm
        y, sr  = librosa.load(path, SR)
        last_label = None

        outfile = open(os.path.join('raw_data/generated/', f.replace('.wav', '.txt')), 'w')
        for i in range(0, len(y), stride):
            if i - last_sound_sample >= min_sound_spacing:                    
                clip = librosa.util.fix_length(y[i:i+onset_increment], onset_increment)
                onset = detect_onset(clip, sr)
                if onset:
                    soundclip = y[i+onset:i+onset+sound_window]
                    start = (i+onset)/sr
                    end = (i + onset + sound_window)/sr
                    label = class_names[np.argmax(predict_one(librosa.util.fix_length(soundclip, sound_window), sr, sound_model))]
                    if not (label=='lip-bass' and last_label=='lip-bass'):
                        last_label = label
                        outfile.write('\t'.join([str(start), str(end), label, '\n']))
                        if not label == 'silence':
                            last_sound_sample = i + onset
                elif last_label == 'kicks':
                    kick_time = .05
                    kick_samples = int(kick_time*sr)
                    soundclip = y[i+kick_samples:i+kick_samples+sound_window]
                    label = class_names[np.argmax(predict_one(librosa.util.fix_length(soundclip, sound_window), sr, sound_model))]
                    if label in ['lip-bass', 'high-hat']:
                        outfile.write('\t'.join([str(start+kick_time), str(end+kick_time), label, '\n']))
                        last_label = label
                        last_sound_sample = i + kick_samples


        outfile.close()
              

if __name__ == '__main__':
    test()