import os
import torch
from torch.autograd import Variable
from model import DeepSpeakerModel
from audio_processing import get_feature

_, _, rawInput = get_feature(os.path.dirname(os.path.abspath(__file__)) + '/s2_n_8_9.wav')

print('Raw input: ', rawInput)

if torch.cuda.is_available():
	tensor = torch.from_numpy(rawInput).type(torch.FloatTensor).cuda()
	cnnModel = DeepSpeakerModel(256, 1).cuda()
else:
	tensor = torch.from_numpy(rawInput).type(torch.FloatTensor)
	cnnModel = DeepSpeakerModel(256, 1)

input = Variable(tensor, requires_grad=True)

print('Input: ', input)

# input = Variable(torch.randn(1, 1, 200, 39).type(torch.FloatTensor), requires_grad=True)

output = cnnModel.forward_classifier(input)

print('OUTPUT: ', output)
