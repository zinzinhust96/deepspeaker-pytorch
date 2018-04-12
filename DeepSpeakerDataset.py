from __future__ import print_function


import numpy as np

import torch.utils.data as data

import os

path = os.path.dirname(os.path.abspath(__file__)) + '/data/BKRecording'

def get_wav_files(path):
    people = []
    classes = []
    for folder in os.listdir(path):
        classes.append(folder)
        f = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".wav"):
                    if (folder in os.path.join(root, file)):
                        f.append(os.path.join(root, file))
        people.append(f)
    return people, classes


def find_classes(voxceleb):
    classes = list(set([datum['speaker_id'] for datum in voxceleb]))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def create_indices(_features):
    inds = dict()
    for idx, (feature_path,label) in enumerate(_features):
        if label not in inds:
            inds[label] = []
        inds[label].append(feature_path)
    return inds


def generate_triplets_call(indices,n_classes):


    # Indices = array of labels and each label is an array of indices
    #indices = create_indices(features)


    c1 = np.random.randint(0, n_classes)
    c2 = np.random.randint(0, n_classes)
    while len(indices[c1]) < 2:
        c1 = np.random.randint(0, n_classes)

    while c1 == c2:
        c2 = np.random.randint(0, n_classes)
    if len(indices[c1]) == 2:  # hack to speed up process
        n1, n2 = 0, 1
    else:
        n1 = np.random.randint(0, len(indices[c1]) - 1)
        n2 = np.random.randint(0, len(indices[c1]) - 1)
        while n1 == n2:
            n2 = np.random.randint(0, len(indices[c1]) - 1)
    if len(indices[c2]) ==1:
        n3 = 0
    else:
        n3 = np.random.randint(0, len(indices[c2]) - 1)


    return ([indices[c1][n1], indices[c1][n2], indices[c2][n3],c1,c2])

class DeepSpeakerDataset(data.Dataset):

    def __init__(self, path, n_triplets,loader, transform=None, *arg, **kw):

        people, classes = get_wav_files(path)
        features = []
        for index, files in enumerate(people):
            for file in files:
                tup = (file, index)
                features.append(tup)
        
        self.root = dir
        #self.features = features
        self.classes = classes
        # self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader

        self.n_triplets = n_triplets

        #print('Generating {} triplets'.format(self.n_triplets))
        self.indices = create_indices(features)



    def __getitem__(self, index):
        '''

        Args:
            index: Index of the triplet or the matches - not of a single feature

        Returns:

        '''
        def transform(feature_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
            """

            feature = self.loader(feature_path)
            return self.transform(feature)

        # Get the index of each feature in the triplet
        a, p, n, c1, c2 = generate_triplets_call(self.indices, len(self.classes))
        # transform features if required
        feature_a, feature_p, feature_n = transform(a), transform(p), transform(n)
        return feature_a, feature_p, feature_n,c1,c2

    def __len__(self):
        return self.n_triplets