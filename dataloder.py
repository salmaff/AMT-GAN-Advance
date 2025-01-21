from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tools.data_reader import DataReader
from backbone.preprocess import PreProcess

class MakeupDataloader(Dataset):
    def __init__(self, image_path, preprocess: PreProcess, transform, transform_mask):
        self.image_path = image_path
        self.transform = transform
        self.transform_mask = transform_mask
        self.preprocess = preprocess
        self.reader = DataReader(image_path)

    def __getitem__(self, index):
        (image_s, mask_s, lm_s), (image_r, mask_r, lm_r) = self.reader.pick()

        # Pastikan lm_s dan lm_r adalah numpy array
        lm_s = np.array(lm_s)
        lm_r = np.array(lm_r)

        # Sesuaikan pembagian dengan ukuran gambar
        lm_s = self.preprocess.relative2absolute(lm_s / np.array([image_s.size[0], image_s.size[1]]))
        lm_r = self.preprocess.relative2absolute(lm_r / np.array([image_r.size[0], image_r.size[1]]))

        image_s = self.transform(image_s)
        mask_s = self.transform_mask(Image.fromarray(mask_s))
        image_r = self.transform(image_r)
        mask_r = self.transform_mask(Image.fromarray(mask_r))

        mask_s, dist_s = self.preprocess.process(mask_s.unsqueeze(0), lm_s)
        mask_r, dist_r = self.preprocess.process(mask_r.unsqueeze(0), lm_r)

        return [image_s, mask_s, dist_s], [image_r, mask_r, dist_r]

    def __len__(self):
        return len(self.reader)

def ToTensor(pic):
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    
    nchannel = {'YCbCr': 3, 'I;16': 1}.get(pic.mode, len(pic.mode))
    img = img.view(pic.size[1], pic.size[0], nchannel).transpose(0, 1).transpose(0, 2).contiguous()
    
    return img.float() if isinstance(img, torch.ByteTensor) else img

def get_loader(config, mode="train"):
    transform = transforms.Compose([
        transforms.Resize(config.DATA.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    transform_mask = transforms.Compose([
        transforms.Resize(config.DATA.IMG_SIZE, interpolation=Image.NEAREST),
        ToTensor
    ])

    dataset = MakeupDataloader(
        config.DATA.PATH, transform=transform,
        transform_mask=transform_mask, preprocess=PreProcess(config)
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True, 
        num_workers=config.DATA.NUM_WORKERS
    )
    return dataloader
