import torch.utils.data as data

from PIL import Image

import os


class ImageFolder(data.Dataset):

    def __init__(self, data_path, transform=None):
        self.path = data_path
        files = os.listdir(self.path)
        
        if len(files) == 0:
            raise(RuntimeError("Found 0 images in: " + self.path + "\n"))
        
        self.files = [os.path.join(self.path,x) for x in files]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        if self.transform is not None:
            img = self.transform(img)
        target = img.convert(mode="L", dither=Image.NONE)

        return img, target



# ds = ImageFolder(data_path='data/')
# for img, target in ds:
#     print(img)
#     img.show()
#     target.show()
#     break