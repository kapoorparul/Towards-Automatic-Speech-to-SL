import numpy as np
import os
import json
import re
import h5py
import librosa
from audio_preprocessing import melspectrogram
import math
import string
import re

'''
This will read the Openpose keypoints, audio and text for every video and save it in the destination folder separately
'''


dst_path = 'data_aud_text'

if not os.path.exists(dst_path):
    print("Directory "+dst_path + " created!")
    os.mkdir(dst_path)

def get_array_with_counter(kp):
    '''
        input shape = x,150 for 3d coordinates
        add counter value at last of every frame keypoints (x,151) and flatten
    '''
    kp = np.array(kp)
    kpts = np.zeros((kp.shape[0], kp.shape[1] + 1))
    full_len = kpts.shape[0]

    for i in range(kp.shape[0]):
    	kpts[i][-1] = i/full_len
    # print(kpts[:][1:].shape, kp.shape)
    for ind,i in enumerate(kp):
    	kpts[ind][:-1] = i
    kpts = kpts.reshape((1, - 1))
    return kpts

def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.
    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i*n:i*n+m]))
        else: # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i*n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)
    
def save_files(dst_path, subset, kpts, specgram, text, name):
    
    ## save skels
    if os.path.exists(os.path.join(dst_path, subset + '.skels')):
        with open(os.path.join(dst_path, subset + '.skels'), "a") as f:
            np.savetxt(f, kpts)
    else:
        np.savetxt(os.path.join(dst_path, subset + ".skels"), kpts)

    ## save audio features
    if os.path.exists(os.path.join(dst_path, subset + '.audio')):
        with open(os.path.join(dst_path, subset + '.audio'), "a") as f:
            np.savetxt(f, specgram)
    else:
        np.savetxt(os.path.join(dst_path, subset + ".audio"), specgram)

    ## save file names
    if os.path.exists(os.path.join(dst_path, subset + '.files')):
        with open(os.path.join(dst_path, subset + '.files'), "a") as f:
            f.write('\n' + name)
    else:
        with open(os.path.join(dst_path, subset + '.files'), "w") as f:
            f.write(name)

    ## save text translation
    if os.path.exists(os.path.join(dst_path, subset + '.text')):
        with open(os.path.join(dst_path, subset + '.text'), "a") as f:
            f.write('\n' + text)
    else:
        with open(os.path.join(dst_path, subset + '.text'), "w") as f:
            f.write(text)



path = '../Data'
all_mean  = 0.22388124641667864
all_std  = 0.4866867249040739

for i, vid in enumerate(sorted(os.listdir(path))):
    
    variations = [n.split('.')[0] for n in sorted(os.listdir(os.path.join(path, vid, 'text')))]
    
    for var in variations:
        audio_path = os.path.join(path, vid, 'audio', var + '.wav')
        text_path = os.path.join(path, vid, 'text', var + '.txt')
        pose_path = os.path.join(path, vid, 'OP',var + '.json')
        if not os.path.exists(pose_path):
            continue
        with open(text_path, 'r') as f:
            text = f.readlines()[0]
        
        text= re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        
        if(text[0]==' '):
            text=text[1:]
        
        if(text[-1]==' '):
            text=text[:-1]

        if text[-1]!='.':
            text += ' .'

        waveform, sr = librosa.load(audio_path, sr = 16000)
        waveform, index = librosa.effects.trim(waveform)  ## to remove starting and trailing silences
        specgram = melspectrogram(waveform).transpose().reshape((1,-1)) ## T, 80
        try:
            with open(pose_path) as f:
                kpts = json.load(f)
        except:
            print("missing/faulty keypoints ", pose_path)
            continue 
        
        kpts = np.array(kpts)
        
        if i<10:
            subset = 'dev'
        elif i<22:
            subset = 'test'
        else:
            subset = 'train'

        try:
            kpts = (kpts - all_mean)/all_std
        except:
            print(kpts_path)
            continue
        
        kpts = kpts / 4
        
        kpts = get_array_with_counter(kpts)
        
        name = vid+'_'+var
        
        save_files(dst_path, subset, kpts, specgram, text, name)
            


