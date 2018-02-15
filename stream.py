import sounddevice as sd
import librosa
import sys

import numpy as np
import sounddevice as sd


from constants import SR, AUDIO_WINDOW_TIME, AUDIO_WINDOW_SAMPLES

from panotti import models as sound_class
from panotti.datautils import *
from predict_class import predict_one

from onsets.predict_class import predict_one as predict_onset
from onsets.panotti import models as onsets
from onsets.constants import ONSET_WINDOW_TIME, ONSET_WINDOW_SAMPLES

global filedata
global sound

class SoundWindow(object):

    def __init__(self, data, blocksize, is_file=False):
        
        self.blocksize = blocksize
        self.is_file = is_file

        if self.is_file:
            self.file_data = data
            self.data = np.ndarray([])
        else:
            self.data = data
        
        self.onset_model = onsets.load_model('onsets/weights.hdf5')
        self.sound_model = sound_class.load_model('weights.hdf5')

        self.last_sound_sample = 0
        self.min_sound_spacing = .06 * SR # minimum resolution: 60 milliseconds =~ 1/32 note at 120bpm
        self.last_label = None

        super(SoundWindow, self).__init__()

    def concat(self, samples):
        if self.is_file:
            data_end = self.data.size
            self.data = np.append(
                self.data, 
                self.file_data[data_end: data_end + self.blocksize]
            )
        else:
            self.data = np.append(self.data, samples)

        return self
        
    @property
    def shape(self):
        return self.data.shape

    def detect_onset(self):
        clip = self.data[-ONSET_WINDOW_SAMPLES:]
        pred = predict_onset(clip, SR, self.onset_model)
        if ['no', 'yes'][np.argmax(pred)] == 'yes':
            ons = librosa.onset.onset_detect(y=clip, sr=sr, units='samples')
            if ons.any():
                return ons[-1]

    def detect_sound(self):



def callback(indata, outdata, frames, time, status):
    global filedata
    global sound
    sound.concat(indata)
    sound_event = sound.detect_sound()



def main(filepath=None):
    global filedata
    global sound

    blocksize = int(44100*.05)

    if filepath:
        data, sr = librosa.load(filepath, sr=SR)
        duration = data.shape[0]/SR
        is_file = True
    else:
        data = np.ndarray([])
        duration = 5
        is_file = False
    
    sound = SoundWindow(data, is_file=is_file, blocksize=blocksize)

    with sd.Stream(channels=1, callback=callback, samplerate = SR, blocksize=blocksize):
        sd.sleep(int(duration * 1000))

if __name__ == '__main__':
    filepath = None
    if len(sys.argv) > 1:
        filepath = sys.argv[-1]
    
    main(filepath=filepath)