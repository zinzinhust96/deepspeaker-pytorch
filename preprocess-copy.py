import re
import os
import bob.ap
import numpy as np
import scipy.io.wavfile as wavfile

def get_bob_extractor(fs, win_length_ms=25, win_shift_ms=10,
                      n_filters=55, n_ceps=15, f_min=0., f_max=6000,
                      delta_win=2, pre_emphasis_coef=0.95, dct_norm=True,
                      mel_scale=True):
    return bob.ap.Ceps(fs, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min,
                      f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)

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
    fs, signal = wavfile.read(filename)
    signal = signal.astype('float64')
    feat = get_bob_extractor(fs, n_filters=40, n_ceps=13)(signal)
    feat = diff_feature(feat, nd=2)
    feat = (feat - np.mean(feat)) / np.std(feat)
    return signal.shape[0], fs, feat.reshape([-1, 39, 1])

