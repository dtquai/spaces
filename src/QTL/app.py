import torch
import torchvision
import gradio as gr
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt

from pennylane import numpy as np
from torchvision import transforms

qubits = 4
batch_size = 8
depth = 6
delta = 0.01

is_cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if is_cuda_available else "cpu")

if is_cuda_available:
    print ("CUDA is available, selected:", device)
else:
    print ("CUDA not available, selected:", device)

dev = qml.device("default.qubit", wires=qubits)

def H_layer(nqubits):
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)

def RY_layer(w):
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def entangling_layer(nqubits):
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat):
    q_weights = q_weights_flat.reshape(depth, qubits)
    H_layer(qubits)
    RY_layer(q_input_features)

    for k in range(depth):
        entangling_layer(qubits)
        RY_layer(q_weights[k])

    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(qubits)]
    return tuple(exp_vals)

class QuantumNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_net = nn.Linear(512, qubits)
        self.q_params = nn.Parameter(delta * torch.randn(depth * qubits))
        self.post_net = nn.Linear(qubits, 2)

    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        q_out = torch.Tensor(0, qubits)
        q_out = q_out.to(device)
        for elem in q_in:
            q_out_elem = quantum_net(elem, self.q_params).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        return self.post_net(q_out)

def classify(image):
    mhModel = torch.load("QKTCC_simPennylane-26032022174332.pth", map_location=device) # huggingface.co/DPT4/quantum-layered-tl
    mMModel = torchvision.models.resnet18(pretrained=True)
    for param in mMModel.parameters():
        param.requires_grad = False
    mMModel.fc = QuantumNet()
    mMModel = mMModel.to(device)
    qModel = mMModel
    qModel.load_state_dict(mhModel)

    from PIL import Image

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    PIL_img = image
    img = data_transforms(PIL_img)
    img_input = img.unsqueeze(0)

    qModel.eval()
    with torch.no_grad():
        outputs = qModel(img_input)
        base_labels = (("mask", outputs[0, 0]), ("no_mask", outputs[0, 1]))
        expvals, preds = torch.max(outputs, 1)
        expvals_min, preds_min = torch.min(outputs, 1)
        if expvals == base_labels[0][1]:
            labels = base_labels[0][0]
        else:
            labels = base_labels[1][0]
        outp = "Classified with output: " + labels + ", Tensor: " + str(expvals) + " (" + str(expvals_min) + ")"
        return outp

out =  gr.outputs.Label(label='Result: ',type='auto')
iface = gr.Interface(classify, gr.inputs.Image(type="pil"), outputs=out,
            title="Quantum Layered TL RN-18 Face Mask Detector",
            description="🤗 This proof-of-concept quantum machine learning model takes a face image input and detects a face that has a mask or no mask: ", theme="default")

iface.launch(debug=True)