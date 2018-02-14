import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from PIL import Image
from panotti.datautils import make_layered_melgram
from multiprocessing import Pool

from constants import AUDIO_WINDOW_SAMPLES, SR


cpu_count = os.cpu_count() - 1
print("",cpu_count,"CPUs detected: Parallel execution across",cpu_count,"CPUs")

LABELS = [
    'doe-singing',
    'growl-bass',
    'high-hat',
    'inward-k-snare',
    'kicks',
    'lip-bass',
    'outward-k-snare',
    'warm-tone-bass',
]

SHORT_SOUNDS = [
    'kicks',
    'high-hat',
    'outward-k-snare',
    'inward-k-snare'
]

NO_SILENCE = ['lip-bass', 'warm-tone-bass']

minimum_relavant_length = .05

def output_clip(data, sr, filepath):

    outpath = os.path.dirname(filepath)
    os.makedirs(outpath, exist_ok=True)
    librosa.output.write_wav(filepath, data, sr)
    # output_image(data, sr, filepath)

def output_image(clip, sr, path):
    plt.subplot(2, 1, 1)
    clip = make_layered_melgram(clip, sr)
    arr = np.reshape(clip, (clip.shape[2],clip.shape[3]))
    librosa.display.specshow(arr, x_axis='time', sr=sr)
    outfile = path.replace('Samples', 'Images').replace('.wav', '.png')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def output_file_sample_range(data, source_filename, start_sample, end_sample, sr, window_size, label, global_start_sample):
    # print ('generating', source_filename, start_sample, end_sample, label, window_size, global_start_sample)
    if window_size > data.shape[0]:
        data = librosa.util.fix_length(data, window_size) #make sure it's at least the needed length

    end = np.int(end_sample - start_sample)
    if label in SHORT_SOUNDS:
        end = window_size
    clip = data[0: end]
    
    # print (clip.shape, 'starting at', global_start_sample)
    if (len(clip)) > window_size:
        i = start_sample
        while i <= end_sample:
            output_file_sample_range(clip[i: i + window_size], source_filename, 0, window_size, sr, window_size, label, global_start_sample + i)
            i += window_size
        return
    
    start = (global_start_sample + start_sample)/sr
    end = (global_start_sample + end_sample)/sr
    
    if label == 'silence':
        if np.amax(clip) > .1:
            return
    else:
        if np.amax(clip) < .1:
            return

    outfilepath = 'Samples/%s/%s-%s-%s.wav'%(label, os.path.basename(source_filename).replace(' ','-'), ('%.3f' % start), ('%.3f' % end))
    output_clip(clip, sr, outfilepath)


class LabelMaker(object):


    def __init__(self, filepath, f, sr, window_size):
        self.sr = np.int(sr)
        self.window_size = window_size

        self.random = self.window_size / 8 / SR
        
        labelpath = filepath.replace('.wav', '.txt')
        labellines = open(labelpath, 'r').readlines()
        self.render_silence = False

        fileLabel = None
        for l in LABELS:
            if f.startswith(l) and l not in NO_SILENCE:
                fileLabel = l
                self.render_silence = True

        self.label_lookup = {
            'start': {},
            'end': {},
            'name': {}
        }
        self.ordered = []
        for l in labellines:
            s, e, tag = l.split('\t')
            start = np.int(np.float(s.strip())*self.sr)
            end = np.int(np.float(e.strip())*self.sr)
            name = tag.strip() or fileLabel
            self.ordered.append((start, end))
            if not name in LABELS:
                name = fileLabel

            self.label_lookup['start'][start] = {'start': start, 'end': end, 'name': name}
            self.label_lookup['end'][end] = {'start': start, 'end': end, 'name': name}
            self.label_lookup['name'][name] = {'start': start, 'end': end, 'name': name}

    def get_label(self, sample_time):
        
        for k, v in self.label_lookup['start'].items():
            if k < sample_time and v['end'] > sample_time:
                # randomize the start a tad
                random = abs(np.random.normal(self.random, self.random))
                start = int(v['start'] - random) 
                return v['name'], start, v['end']
        # if the next label starts before the end of this current window, don't make a silence clip

        labels_after = [s for s, e in self.ordered if s > sample_time]
        if labels_after:
            if labels_after[0] - sample_time < self.window_size:
                return None, None, None
        
        return 'silence', None, None

def handle_file(f):
    print ("doing", f)
    render_silence = False
    filepath = os.path.join('raw_data', f)
    y, sr  = librosa.load(filepath, SR)

    window_samples = AUDIO_WINDOW_SAMPLES
    labelMaker = LabelMaker(filepath, f, sr, window_samples)
    
    i = 0
    while i < len(y):
        start = i/SR
        end = (i + window_samples)/SR
        clip = librosa.util.fix_length(y[i:i+window_samples], window_samples)
        outfile = 'Samples/silence/%s-%s-%s.wav'%(f.replace(' ','-'), str(start).replace('.', '_'), str(end).replace('.', '_'))
        label = None

        label, labelstart, labelend = labelMaker.get_label(i)

        if label in LABELS:
            
            output_file_sample_range(
                y[labelstart : labelend], # the full range of the label
                filepath, #source file,
                0, #start sample,
                labelend - labelstart, # end sample
                sr,
                window_samples, # increment samples,
                label,
                labelstart #global start sample
            )

        # output this window interval of the original source 
        if label == 'silence' and labelMaker.render_silence:
            # print ('arbitrary range', i)
            output_file_sample_range(
                    y[i:i+window_samples], # the file
                    filepath , #source file,
                    0, #start sample,
                    window_samples, # end sample
                    sr,
                    window_samples, # increment samples
                    label,
                    i # the current sample in the file
                ) 

        i += window_samples
    print ("finished", f)

def process_rawdata():
    for d in os.listdir('Samples'):
        os.rmdir(os.path.join('Samples', d))

    for d in os.listdir('Preproc/Train'):
        os.rmdir(os.path.join('Preproc/Train', d))
    
    for d in os.listdir('Preproc/Test'):
        os.rmdir(os.path.join('Preproc/Test', d))
    
    files =  [f for f in os.listdir('raw_data') if f.endswith('.wav')]
    pool = Pool(cpu_count)
    pool.map(handle_file, files)


if __name__ == '__main__':
    process_rawdata()