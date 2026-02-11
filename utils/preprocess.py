import torch
import torchvision.transforms as T

def preprocess_frames(frames):
    transform = T.Compose([
        T.Resize((112, 112)),
        T.CenterCrop(112),
        T.ToTensor()
    ])
    frames_tensor = torch.stack([transform(frame) for frame in frames])
    frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (B, C, T, H, W)
    return frames_tensor
