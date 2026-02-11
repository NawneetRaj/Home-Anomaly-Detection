import torch
from torchvision.models.video import r3d_18
from torchvision import transforms
import numpy as np

class BehaviorRecognizer:
    def __init__(self):
        # Load pretrained model
        self.model = r3d_18(pretrained=True)
        self.model.eval()

        # Define transformation for the input frame
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def recognize(self, frame):
        # Convert the frame from numpy array to PyTorch tensor
        frame_tensor = self.transform(frame).unsqueeze(0).float()  # Add batch dimension
        
        # Ensure the frame tensor is on the right device (CPU or GPU)
        if torch.cuda.is_available():
            frame_tensor = frame_tensor.cuda()
            self.model = self.model.cuda()
        
        # Run the model to get predictions
        output = self.model(frame_tensor)
        
        # Get the predicted class
        pred_class = output.argmax(dim=1)
        return pred_class.item()
