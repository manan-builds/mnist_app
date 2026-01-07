from torchvision import transforms

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((32, 32),
        transforms.ToTensor(),
        transforms.Normalize(
          mean=(0.5, 0.5, 0.5),
          std=(0.5, 0.5, 0.5))])
        