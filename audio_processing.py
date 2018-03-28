import re
import os
import bob.ap
import numpy as np
import scipy.io.wavfile as wavfile

def diff_feature(feat, nd=1):
    diff = feat[1:] - feat[:-1]
    feat = feat[1:]
    if nd == 1:
        return np.concatenate((feat, diff), axis=1)
    if nd == 2:
        d2 = diff[1:] - diff[:-1]
        return np.concatenate((feat[1:], diff[1:], d2), axis=1)
    return -1

def get_feature(filename):
    sample_rate, signal = wavfile.read(filename)
    signal = signal.astype('float64')
    feat = bob.ap.Ceps(sample_rate, win_length_ms=25, win_shift_ms=10,
                      n_filters=40, n_ceps=13, f_min=20., f_max=7600,
                      delta_win=2, pre_emphasis_coeff=0.96, dct_norm=True,
                      mel_scale=True)(signal)
    feat = diff_feature(feat, nd=2)
    feat = (feat - np.mean(feat)) / np.std(feat)
    return signal.shape[0], sample_rate, feat.reshape([-1, 39, 1])

def main():
    print(get_feature('/home/zinzin/Documents/pytorch/deepspeaker-pytorch/s2_n_8_9.wav')[2].shape)

if __name__ == '__main__':
    main()