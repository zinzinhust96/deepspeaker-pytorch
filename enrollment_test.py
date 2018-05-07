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
from files import readEnrollmentPaths, readTestPaths

file_transform = transforms.Compose([
                truncatedinput(),
                toMFB(),
                totensor(),
                #tonormal()
            ])
file_loader = read_audio

l2_dist = PairwiseDistance(2)

model = DeepSpeakerModel(embedding_size=256,
                      num_classes=3)

def transform(feature_path):
    """Convert image into numpy array and apply transformation
        Doing this so that it is consistent with all other datasets
    """
    feature = file_loader(feature_path)
    return file_transform(feature)

def calculateOneEmbedding(path):
    feature = transform(path)
    feature = Variable(feature.unsqueeze(0))  
    return model(feature)

def enrollment(model):
    indices, classes = readEnrollmentPaths(os.path.dirname(os.path.abspath(__file__)) + '/data/test_data')

    embeddings = {}
    for key, value in indices.items():
        embeddings[key] = Variable(torch.FloatTensor(1, 256).zero_())
        numberOfUtterance = len(value)
        for path in value:
            out = calculateOneEmbedding(path)
            embeddings[key] += out
        embeddings[key] /= numberOfUtterance

    return embeddings

def test(model, embeddings):
    labels, results = [], []
    
    indices, classes = readTestPaths(os.path.dirname(os.path.abspath(__file__)) + '/data/test_data')

    for key, value in indices.items():
        for path in value:
            labels.append(key)            
            out = calculateOneEmbedding(path)
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
    embeddings = enrollment(model)
    test(model, embeddings)


if __name__ == '__main__':
    main()