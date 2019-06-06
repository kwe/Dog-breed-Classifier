import torch
import torch.onnx
import torchvision.models as models
import torch.nn as nn

model = models.vgg16(pretrained=True)

for param in model.features.parameters():
  param.requires_grad = False

num_inputs = model.classifier[6].in_features

last_layer = nn.Linear(num_inputs, 133)

model.classifier[6] = last_layer

print(model)

state_dict = torch.load('model_transfer.pt')
model.load_state_dict(state_dict)

dummy_input = torch.randn(1,3,244,244)

torch.onnx.export(model, dummy_input, "onnx_model.onnx", verbose=True)
