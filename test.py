from srgan_module import SRGAN
from cyclegan import CYCLEGAN
from srresnet_module import SRResNet
from pytorch_lightning.utilities.model_summary import ModelSummary
from PIL import Image
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from skimage.io import imread, imsave
import glob
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
# from save_as_png import load_tiff_stack
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif'
]


def is_image(path: Path):
    return path.suffix in IMG_EXTENSIONS


lr_transforms = transforms.Compose(
    [

        transforms.Normalize(mean=(-1.0,) * 1, std=(2.0,) * 1),
        transforms.ToPILImage(),
        transforms.Resize(512, Image.BICUBIC),
        transforms.ToTensor(),
    ]
)


def load_tiff_stack(file):
    """ :paramfile: Path object describing the location of the file :return: a numpy array of the volume """

    if not (file.name.endswith('.tif') or file.name.endswith('.tiff')):
        raise FileNotFoundError('Input has to be tif.')
    else:
        volume = imread(file, plugin='tifffile')
        return volume


def save_to_tiff_stack(array, file):
    """ :paramarray: Array to save :paramfile: Path object to save to """

    if not file.parent.is_dir():
        file.parent.mkdir(parents=True, exist_ok=True)
    if not (file.name.endswith('.tif') or file.name.endswith('.tiff')):
        raise FileNotFoundError('File has to be tif.')
    else:
        imsave(file, array, plugin='tifffile', check_contrast=False)


# LOAD MODEL

model = CYCLEGAN.load_from_checkpoint(
    checkpoint_path='lightning_logs/cyclegan/train/checkpoints/epoch=103-step=7904.ckpt')
# model = SRResNet.load_from_checkpoint(
#     checkpoint_path="lightning_logs/srresnet/stl10-scale_factor=4/checkpoints/epoch=49-step=3550.ckpt", image_channels =1)
print(ModelSummary(model))
# image = Image.open('4.png')
# transform = transforms.Compose([transforms.ToTensor()])
# lr = transform(image)
# lr = lr[None, :, :, :]


data = '/home/woody/iwal/iwal052h/PR_SRGAN/Test/'
# tif = './ERC93_042.tif'
# op = 'D:/FAU/SEM5_FAU/PR_SRGAN/SRGan_Res/'
# if not os.path.exists(op):
#     os.makedirs(op)
dir = Path(data)

# dir = Path(tif)
# tif_data = load_tiff_stack(dir)

# filenames = [f for f in dir.glob('*') if is_image(f)]

f = [f for f in dir.glob('*') if is_image(f)]
c = 1
print(len(f))
for i in f:

    vol = load_tiff_stack(i)
    tensors = []
    for j in vol:
        v = torch.from_numpy(j).unsqueeze(0)
        # v = v[None, :, :]   #.unsqueeze(dim=0)
        v = lr_transforms(v)
        # v = v[None, :, :]
        v = v.unsqueeze(0)
        with torch.no_grad():
            sr = model(v)
        op = sr.cpu().detach().numpy()
        file = 'SR_output/' + str(c) + '.tif'
        file_path = Path(file)
        save_to_tiff_stack(op, file_path)
        c += 1

# FOR PNG DATA

# for file in filenames:
#     image = Image.open(file)
#
#     lr = transform(image)
#     lr = lr[None, :, :, :]
#
#     sr = model(lr)
#     save_image(sr, op + file.stem + '.png')

# FOR TIF DATA

# for d in tif_data:
#     print(np.max(d))
#     print(np.min(d))
#     d = (d - np.amin(d)) / (np.amax(d) - np.amin(d))
#     print(np.max(d))
#     print(np.min(d))
#     im = Image.fromarray(np.uint8(d*255))
#     im.show()
#     im = transform(im)
#     print(im.size())
#     break


# sr = model(lr)
# save_image(sr, '4_final.png')

