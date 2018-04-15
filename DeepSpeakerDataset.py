from __future__ import print_function


import numpy as np
import torch.utils.data as data
import os

def create_indices(_path):
    """Returns 2 items: 1 dict contains the arrays of path to the wav files for each person,
    and 1 array of class names
    """
    indices = {}
    classes = []
    for label in os.listdir(_path):
        classes.append(label)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    for label in os.listdir(_path):
        indices[class_to_idx[label]] = []
        for root, dirs, files in os.walk(_path):
            for file in files:
                if file.endswith(".wav"):
                    if (label in os.path.join(root, file)):
                        indices[class_to_idx[label]].append(os.path.join(root, file))
    return indices, classes


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

        indices, classes = create_indices(path)
        
        self.root = dir
        #self.features = features
        self.classes = classes
        # self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader

        self.n_triplets = n_triplets

        #print('Generating {} triplets'.format(self.n_triplets))
        self.indices = indices



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