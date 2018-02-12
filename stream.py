import sounddevice as sd
import librosa
import sys

import numpy as np
import sounddevice as sd

global filedata
global sound


class SoundWindow(np.ndarray):


    def __new__(cls, inputarr):
        obj = np.asarray(inputarr).view(cls)
        return obj

    def __init__(self, *args, **kwargs):
        super(SoundWindow, self).__init__()

    def concat(self, samples):

        print (self.shape, samples.shape)
        # return np.concatenate(self, samples)


def callback(indata, outdata, frames, time, status):
    global filedata
    global sound

    sound.concat(indata)


def main(filepath):
    global filedata
    global sound

    filedata, sr = librosa.load(filepath, sr=44100)
    duration = filedata.shape[0]/sr

    sound = SoundWindow((0,1))

    with sd.Stream(channels=1, callback=callback, samplerate = 44100, blocksize=int(44100/20)):
        sd.sleep(int(duration * 1000))


if __name__ == '__main__':
    filepath = sys.argv[-1]
    main(filepath)