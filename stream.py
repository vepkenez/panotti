import sounddevice as sd
import librosa
import sys
import time

import numpy as np
import sounddevice as sd

from constants import SR, AUDIO_WINDOW_TIME, AUDIO_WINDOW_SAMPLES

from panotti import models as sound_class
from panotti.datautils import get_class_names as get_audio_classes
from predict_class import predict_one as classify_audio

from onsets.predict_class import predict_one as predict_onset
from onsets.panotti import models as onsets
from onsets.constants import ONSET_WINDOW_TIME, ONSET_WINDOW_SAMPLES

global filedata
global sound
global start

start = None

onset_model = onsets.load_model('onsets/weights.hdf5')
sound_model = sound_class.load_model('weights.hdf5')


class SoundWindow(object):

    def __init__(self, data, blocksize, is_file=False):
        
        self.blocksize = blocksize
        self.is_file = is_file

        if self.is_file:
            self.file_data = data
            self.data = np.ndarray([])
        else:
            self.data = data

        self.last_sound_sample = 0
        self.min_sound_spacing = .06 * SR # minimum resolution: 60 milliseconds =~ 1/32 note at 120bpm

        self.last_label = None
        self.class_names = get_audio_classes()

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

    @property
    def size(self):
        return self.data.size

    def detect_onset(self):
        self.sound_clip = self.data[-AUDIO_WINDOW_SAMPLES:]
        clip = self.sound_clip[0: ONSET_WINDOW_SAMPLES]
        clip = librosa.util.fix_length(
                            clip, 
                            ONSET_WINDOW_SAMPLES
                        )
        pred = predict_onset(clip, SR, onset_model)
        if ['no', 'yes'][np.argmax(pred)] == 'yes':
            ons = librosa.onset.onset_detect(y=clip, sr=SR, units='samples')
            if ons.any():
                return ons[0]

    def detect_sound(self, current_time):
        time_stamp = current_time / SR
        print (time_stamp)
        if time_stamp - self.last_sound_sample > self.min_sound_spacing:
            onset = self.detect_onset()
            soundclip = self.sound_clip
            if onset:
                soundclip = soundclip[onset:]
                label = self.class_names[
                    np.argmax(
                        classify_audio(
                            librosa.util.fix_length(
                                soundclip, 
                                AUDIO_WINDOW_SAMPLES
                            ), 
                            SR, 
                            sound_model
                        )
                    )
                ]
                print (time_stamp - self.last_sound_sample)
                self.last_sound_sample = time_stamp
                return label, onset

        return None, None
    




def callback(indata, outdata, frames, timedata, status):
    global filedata
    global sound
    global start

    if not start:
        start = timedata.currentTime

    sound.concat(indata)
    sound_event, onset = sound.detect_sound(timedata.currentTime-start)
    if sound_event:
        print (sound_event, (sound.size-AUDIO_WINDOW_TIME)/SR)


def main(filepath=None):
    global filedata
    global sound
    global start

    blocksize = int(44100*.05)

    if filepath:
        data, sr = librosa.load(filepath, sr=SR)
        duration = data.shape[0]/SR
        is_file = True
       

    else:
        data = np.ndarray([])
        duration = 10
        is_file = False

    # warm up the models        
    predict_onset(np.zeros(ONSET_WINDOW_SAMPLES), SR, onset_model)
    classify_audio(np.zeros(AUDIO_WINDOW_SAMPLES), SR, sound_model)
    
    sound = SoundWindow(data, is_file=is_file, blocksize=blocksize)

    print ("streaming")
    with sd.Stream(channels=1, callback=callback, samplerate = SR, blocksize=blocksize):
        sd.sleep(int(duration * 1000))

if __name__ == '__main__':
    filepath = None
    if len(sys.argv) > 1:
        filepath = sys.argv[-1]
    
    main(filepath=filepath)