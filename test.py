import torch
from torch.autograd import Variable
from model import DeepSpeakerModel

cnnModel = DeepSpeakerModel(512, 1)

input = Variable(torch.randn(1, 1, 200, 39).type(torch.FloatTensor), requires_grad=True)

output = cnnModel.forward_classifier(input)

print(output)
