import pathlib
import sys
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import OCTDataset

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        gaussian_blur_ks = size[0] if isinstance(size, tuple) else size
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * gaussian_blur_ks)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views, oct_args=None):
        # Add support for OCT dataset
        if oct_args is not None:
            if oct_args['use_simclr_augmentations']:
                data_transforms = ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(oct_args['img_size']),
                    n_views).base_transform.transforms

                # Append data_transforms to base transforms
                img_transforms = oct_args['transforms_list']
                if oct_args['img_channel'] == 1:
                    img_transforms = img_transforms[:-1]  # Remove Grayscale transform, will be re-added later
                data_transforms = img_transforms + data_transforms[:-1]  # Removing second ToTensor
                if oct_args['img_channel'] == 1:
                    data_transforms.append(transforms.Grayscale())  # Re-add Grayscale transform at the end
                oct_args['transforms_list'] = data_transforms

            # Convert to Compose
            oct_args['transforms'] = transforms.Compose(oct_args['transforms_list'])
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),
                          'oct': lambda: OCTDataset(self.root_folder, 'train',
                                                    oct_args['map_df_paths'], oct_args['labels_dict'],
                                                    ch_in=oct_args['img_channel'],
                                                    sample_within_image=oct_args['sample_within_image'],
                                                    use_iipp=oct_args['use_iipp'],
                                                    num_same_area=oct_args['num_same_area'],
                                                    transforms=ContrastiveLearningViewGenerator(
                                                        oct_args['transforms'], n_views),
                                                    pre_sample=oct_args['dataset_sample']),
                          }
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
