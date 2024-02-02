import torch
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a pre-trained model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Load an image and preprocess it
image_path = '/frame_0001.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_image = preprocess(image)
input_image = input_image.unsqueeze(0)  # Add batch dimension

# Get the prediction for the class of interest (e.g., top-1 prediction)
with torch.no_grad():
    output = model(input_image)

class_idx = torch.argmax(output).item()

# Define the model for Grad-CAM
class ModelWithHooks:
    def __init__(self, model):
        self.model = model
        self.features = None

        self.hook = self.model.layer4.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def forward(self, x):
        return self.model(x)

model_with_hooks = ModelWithHooks(model)

# Compute the gradient with respect to the predicted class
model_with_hooks.forward(input_image)

# Ensure that the feature_map requires gradients
feature_map = model_with_hooks.features
feature_map.requires_grad_()

output[0, class_idx].backward()

# Get the gradients
gradients = feature_map.grad

# Compute Grad-CAM
alpha = gradients.mean(dim=(2, 3), keepdim=True)
weighted_feature_map = (feature_map * alpha).sum(dim=1, keepdim=True)
grad_cam = torch.nn.functional.relu(weighted_feature_map)

# Resize Grad-CAM to match the original image size
grad_cam = torch.nn.functional.interpolate(grad_cam, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)

# Convert Grad-CAM to a numpy array
grad_cam = grad_cam[0].cpu().numpy()
grad_cam = np.maximum(grad_cam, 0)
grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam))

# Overlay the Grad-CAM on the original image
heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
output_image = heatmap * 0.5 + image

# Display the original image with Grad-CAM
plt.imshow(output_image)
plt.axis('off')
plt.show()
