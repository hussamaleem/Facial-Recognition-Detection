from torchvision import transforms,datasets
from torch.utils.data import random_split
import utils

def create_dataset(train,face_detection):
    
    dataset_path = utils.get_path(train=train,
                                  face_detection=face_detection)
    
    transform = transforms.Compose([
        transforms.Resize((80,80)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    dataset = datasets.ImageFolder(root=dataset_path,
                                   transform=transform)
    
    if face_detection:
        
        train_size = int(0.8*len(dataset))
        test_size = len(dataset)-train_size
        
        train_dataset,test_dataset = random_split(dataset, [train_size,test_size])
        
        return train_dataset,test_dataset
    
    else:    
        return dataset