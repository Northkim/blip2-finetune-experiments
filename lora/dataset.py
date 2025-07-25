import os
from PIL import Image
from torch.utils.data import Dataset

class HMIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # 遍历所有类别文件夹
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        fpath = os.path.join(class_dir, fname)
                        caption = class_name.replace('_', ' ')  # 类名作为 caption
                        self.samples.append((fpath, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, caption
