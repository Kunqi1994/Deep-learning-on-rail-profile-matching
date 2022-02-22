import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class KQdata(Dataset):
    def __init__(self, fold, transform=None):
        self.fold = fold
        self.path = os.listdir(self.fold)
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.path[idx]
        image = Image.open(os.path.join(self.fold, image_path, 'sample1.png'))

        if self.transform is not None:
            image = self.transform(image)

        f_label = open(os.path.join(self.fold, image_path, 'annotations_file.txt'))
        label = f_label.readline()
        label = label.split('	')
        label = list(map(float, label))
        label = torch.Tensor(label)

        return image, label

    def __len__(self):
        return len(self.path)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose((transforms.Resize(224),
                                transforms.ToTensor(),
                                normalize))

train_dataset = KQdata(fold='train',
                       transform=transform)

validation_dataset = KQdata(fold='val',
                            transform=transform)

test_dataset = KQdata(fold='test',
                      transform=transform)


train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=0)

validation_loader = DataLoader(validation_dataset,
                               batch_size=32,
                               shuffle=True,
                               num_workers=0)

test_loader = DataLoader(test_dataset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=0)
