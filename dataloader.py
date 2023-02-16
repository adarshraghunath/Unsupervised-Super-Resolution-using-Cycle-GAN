from pathlib import Path
import functools
import numpy as np
from skimage.io import imread, imsave
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transform_lib
from torchvision.transforms import functional as TF

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif'
]


def is_image(path: Path):
    return path.suffix in IMG_EXTENSIONS


def pad(img, scale):
    width, height = img.size
    pad_h = width % scale
    pad_v = height % scale
    img = TF.pad(img, (0, 0, scale - pad_h, scale - pad_v), padding_mode='reflect')
    return img


def load_tiff_stack(file):
    """ :paramfile: Path object describing the location of the file :return: a numpy array of the volume """

    if not (file.name.endswith('.tif') or file.name.endswith('.tiff')):
        raise FileNotFoundError('Input has to be tif.')
    else:
        volume = imread(file, plugin='tifffile')
        return volume

def normalize_data_in_window(x, w_min, w_max):
    '''Maps data in range [w_min, w_max] to [0, 1]. Clipping of data to zero or one for values outside of [w_min, w_max]. :paramx: Input image. :paramw_min: Lower bound of window. :paramw_max: Upper bound of window. :return: Normalized tensor. '''
    x_norm = (x - w_min) / (w_max - w_min)
    x_norm[x_norm >= 1.0] = 1.0

    x_norm[x_norm <= 0.0] = 0.0

    return x_norm

class DatasetFromFolder(Dataset):
    def __init__(self, data_dir, scale_factor, image_type: str, image_channels: int, patch_size=256, preupsample=False):
        assert patch_size % scale_factor == 0
        self.patch_size = patch_size

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        self.image_type = image_type
        if self.image_type == '.tif':
            filenames = []
            f = [f for f in data_dir.glob('*') if is_image(f)]
            for i in f:
                vol = load_tiff_stack(i)
                filenames.append(vol)
            arr = np.vstack(filenames)

            self.filenames = arr

        else:
            self.filenames = [f for f in data_dir.glob('*') if is_image(f)]

        self.scale_factor = scale_factor
        self.preupsample = preupsample
        hr_image_size = self.patch_size
        lr_image_size = hr_image_size // scale_factor
        self.unnormalized_hr = transform_lib.Compose([transform_lib.CenterCrop(hr_image_size), transform_lib.ToTensor()])
        self.hr_transforms = transform_lib.Compose(
            [
                # transform_lib.Grayscale(num_output_channels=1),
                transform_lib.RandomCrop(hr_image_size),
                transform_lib.RandomRotation(degrees=(-90,90)),
                transform_lib.ToTensor(),
                # transform_lib.Normalize(mean=(0.0,) * image_channels, std=(1.0,) * image_channels)


            ]
        )

        self.lr_transforms = transform_lib.Compose(
            [

                transform_lib.Normalize(mean=(-1.0,) * image_channels, std=(2.0,) * image_channels),
                transform_lib.ToPILImage(),
                transform_lib.Resize(lr_image_size, Image.BICUBIC),
                transform_lib.ToTensor(),
            ]
        )

        self._hr_imgs = [normalize_data_in_window(self.hr_transforms(Image.fromarray(img)),-1.0,1.0) for img in self.filenames]

        self._lr_imgs = [self.lr_transforms(img) for img in self._hr_imgs]

        # random.shuffle(self._hr_imgs)
        # random.shuffle(self._lr_imgs)

    def __getitem__(self, index):

        assert len(self.filenames) != 0
        filename = self.filenames[index]

        # img = Image.open(filename).convert('RGB')    #commented this out

        if self.image_type == '.tif':
            img = self.filenames[index]
            img = Image.fromarray(img)

        else:
            img = Image.open(filename)

        # img_hr = self.transforms(img)
        # down_size = [l // self.scale_factor for l in img_hr.size[::-1]]
        # img_lr = TF.resize(img_hr, down_size, interpolation=Image.BICUBIC)
        # if self.preupsample:
        #     img_lr = TF.resize(img_lr, img_hr.size[::-1], interpolation=Image.BICUBIC)
        un_hr = self.unnormalized_hr(img)
        img_hr = self.hr_transforms(img)
        img_lr = self.lr_transforms(img_hr)

        # return {'lr': TF.to_tensor(img_lr), 'hr': TF.to_tensor(img_hr), 'path': filename.stem}
        return un_hr, self._hr_imgs[index],self._lr_imgs[index]

    def __len__(self):
        return len(self.filenames)
