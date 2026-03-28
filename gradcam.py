import torch
import cv2
import numpy as np

def generate_gradcam(model, img_tensor):

    model.eval()

    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Last conv layer (works for both models)
    target_layer = list(model.features.children())[-1]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred = output.argmax()

    model.zero_grad()
    output[0, pred].backward()

    grads = gradients[0]
    fmap = features[0]

    weights = torch.mean(grads, dim=(2,3), keepdim=True)
    cam = torch.sum(weights * fmap, dim=1).squeeze()

    cam = cam.detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = cam / cam.max()

    handle_f.remove()
    handle_b.remove()

    return cam