import torch
from torch.autograd import Variable
from model import DeepSpeakerModel
from audio_processing import get_feature

rawInput = get_feature('/home/zinzin/Documents/pytorch/deepspeaker-pytorch/s2_n_8_9.wav')[2]

print('Raw input: ', rawInput)

input = Variable(torch.from_numpy(rawInput).type(torch.FloatTensor).view(1, 1, -1, 39).cuda(), requires_grad=True)

print('Input: ', input)

cnnModel = DeepSpeakerModel(512, 1)
cnnModel.cuda()

# input = Variable(torch.randn(1, 1, 200, 39).type(torch.FloatTensor), requires_grad=True)

output = cnnModel.forward_classifier(input)

print('OUTPUT: ', output)
