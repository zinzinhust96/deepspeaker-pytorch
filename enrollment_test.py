from __future__ import print_function

import torch
import numpy as np
import torch.utils.data as data
import os
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import DeepSpeakerModel
from audio_processing import toMFB, totensor, truncatedinput, tonormal, truncatedinputfromMFB,read_MFB,read_audio,mk_MFB
from utils import PairwiseDistance

file_transform = transforms.Compose([
                truncatedinput(),
                toMFB(),
                totensor(),
                #tonormal()
            ])
file_loader = read_audio

l2_dist = PairwiseDistance(2)

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

def transform(feature_path):
    """Convert image into numpy array and apply transformation
        Doing this so that it is consistent with all other datasets
    """
    feature = file_loader(feature_path)
    return file_transform(feature)

def enrollment(model):
    indices, classes = create_indices(os.path.dirname(os.path.abspath(__file__)) + '/data/enrollment_set')

    embeddings = {}
    for key, value in indices.items():
        embeddings[key] = Variable(torch.FloatTensor(1, 256).zero_())
        numberOfUtterance = len(value)
        for path in value:
            feature = transform(path)
            feature = Variable(feature.unsqueeze(0))  
            out = model(feature)
            embeddings[key] += out
        embeddings[key] /= numberOfUtterance

    return embeddings

def test(model, embeddings):
    labels, results = [], []
    
    indices, classes = create_indices(os.path.dirname(os.path.abspath(__file__)) + '/data/test_set')

    for key, value in indices.items():
        for path in value:
            labels.append(key)            
            feature = transform(path)
            feature = Variable(feature.unsqueeze(0))  
            out = model(feature)
            distances = []            
            for key_e, value_e in embeddings.items():
                dists = l2_dist.forward(out,value_e)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
                distances.append(dists.data.cpu().numpy())
            # print('DISTANCE: ', distances)
            result = np.argmin(distances)
            # print('RESULT: ', result)
            results.append(result)

    print('LABELS: ', labels)
    print('RESULTS: ', results)

    accuracy = sum(1 for x,y in zip(labels,results) if x == y) / len(labels)

    print('ACCURACY: ', accuracy)

    return 0


def main():
    model = DeepSpeakerModel(embedding_size=256,
                      num_classes=3)
    
    embeddings = enrollment(model)
    test(model, embeddings)




if __name__ == '__main__':
    main()