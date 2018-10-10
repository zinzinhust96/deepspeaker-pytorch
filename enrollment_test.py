from __future__ import print_function

import argparse
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
from constants import TEST_DIR

_path = TEST_DIR

parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition Enrollment and Test')
parser.add_argument('--resume',
                    default=None,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()

file_transform = transforms.Compose([
                truncatedinput(),
                toMFB(),
                totensor(),
                #tonormal()
            ])
file_loader = read_audio

l2_dist = PairwiseDistance(2)

def transform(feature_path):
    """Convert image into numpy array and apply transformation
        Doing this so that it is consistent with all other datasets
    """
    feature = file_loader(feature_path)
    return file_transform(feature)

def calculateOneEmbedding(path, model):
    feature = transform(path)
    feature = Variable(feature.unsqueeze(0))
    return model(feature)

def enrollment(model):
    indices, classes = readEnrollmentPaths(_path)

    embeddings = {}
    for key, value in indices.items():
        embeddings[key] = Variable(torch.FloatTensor(1, 256).zero_())
        numberOfUtterance = len(value)
        for path in value:
            out = calculateOneEmbedding(path, model)
            embeddings[key] += out
        embeddings[key] /= numberOfUtterance
    
    # print('embeddings: ', embeddings)

    return embeddings

def test(model, embeddings):
    labels, results = [], []
    
    indices, classes = readTestPaths(_path)

    for key, value in indices.items():
        for path in value:
            labels.append(key)            
            out = calculateOneEmbedding(path, model)
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
                      num_classes=10)

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    # print(calculateOneEmbedding('/home/zinzin/Documents/pytorch/deepspeaker-pytorch/data/test_set/dnl/s1/t1/s1_t1_1.wav', model))
    
    embeddings = enrollment(model)
    test(model, embeddings)


if __name__ == '__main__':
    main()