import abc
from random import shuffle

import numpy as np

import bids
import nibabel as nib


class bSSFPDatasetGenerator:
    def __init__(self,
                 files,
                 shape,
                 augment=False,
                 aug_fact=1,
                 seed=42):
        self.filenames = files
        self.in_shape = shape
        self.num_files = len(files)
        self.seed = seed
        self.augment = augment
        self.aug_fact = aug_fact

    def __len__(self):
        return self.num_files * self.aug_fact

    def __call__(self):
        for i in range(len(self)):
            yield self.read_nifti_file(i)

    def z_normalize_img(self, img):
        """
        Normalize the image so that the mean value for each image
        is 0 and the standard deviation is 1.
        """
        for channel in range(img.shape[-1]):

            img_temp = img[..., channel]
            img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)

            img[..., channel] = img_temp

        return img

    def augment_data(self, img, omap):
        """
        Data augmentation
        Flip image and rotate image.
        """

        # Determine if axes are equal and can be rotated
        # If the axes aren't equal then we can't rotate them.
        equal_dim_axis = []
        for idx in range(0, len(img.shape)):
            for jdx in range(idx+1, len(img.shape)):
                if img.shape[idx] == img.shape[jdx]:
                    equal_dim_axis.append([idx, jdx])  # Valid rotation axes
        dim_to_rotate = equal_dim_axis

        if np.random.rand() > 0.5:
            # Random 0,1 (axes to flip)
            ax = np.random.choice(np.arange(len(img.shape)-1))
            img = np.flip(img, ax)
            omap = np.flip(omap, ax)

        if (len(dim_to_rotate) > 0) and (np.random.rand() > 0.5):
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

            # This will choose the axes to rotate
            # Axes must be equal in size
            random_axis = dim_to_rotate[np.random.choice(len(dim_to_rotate))]

            img = np.rot90(img, rot, axes=random_axis)  # Rotate axes 0 and 1
            omap = np.rot90(omap, rot, axes=random_axis)  # Rotate axes 0 and 1

        return img, omap

    def read_nifti_file(self, idx):
        """
        Read Nifti file
        """
        f_idx = idx % self.num_files
        imgFile = self.filenames[f_idx][0]
        omapFile = self.filenames[f_idx][1]

        img = np.array(nib.load(imgFile).dataobj)
        img = np.stack([np.abs(img), np.angle(img)], axis=-1)
        img = np.reshape(img, self.in_shape)

        omap = np.array(nib.load(omapFile).dataobj)

        if self.augment:
            img, omap = self.augment_data(img, omap)

        # Normalize
        img = self.z_normalize_img(img)

        return img, omap

    def plot_images(self, ds, slice_num=50):
        """
        Plot images from dataset
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 20))
        num_cols = 2
        img_channel = 0

        for img, omap in ds.take(1):
            bs = img.shape[0]

            for idx in range(bs):
                plt.subplot(bs, num_cols, idx*num_cols + 1)
                plt.imshow(img[idx, slice_num, :, :, img_channel], cmap="bone")
                plt.title("MRI", fontsize=18)
                plt.subplot(bs, num_cols, idx*num_cols + 2)
                plt.imshow(omap[idx, :, :, slice_num, 0], cmap="bone")
                plt.title("MRI", fontsize=18)

        plt.show()


class bSSFPBaseDatasetLoader(abc.ABC):
    def __init__(self,
                 data_path,
                 aug_fact,
                 train_test_split,
                 validate_test_split,
                 random_seed):
        self.data_path = data_path
        self.train_test_split = train_test_split
        self.validate_test_split = validate_test_split
        self.random_seed = random_seed

        self.bids_layout = bids.BIDSLayout(
                data_path,
                validate=False,
                database_path=data_path + '/dove.db')
        self.bids_layout.add_derivatives(
                data_path + '/derivatives/preproc-svenja',
                database_path=data_path + '/preproc-svenja.db')
        self.bids_layout.add_derivatives(
                data_path + '/derivatives/preproc-dove',
                database_path=data_path + '/preproc-dove.db')

        self.create_file_list()

        self.train_generator = bSSFPDatasetGenerator(self.train_filenames,
                                                     self.in_shape,
                                                     augment=True,
                                                     aug_fact=aug_fact)
        self.val_generator = bSSFPDatasetGenerator(self.val_filenames,
                                                   self.in_shape)
        self.test_generator = bSSFPDatasetGenerator(self.test_filenames,
                                                    self.in_shape)

    def get_generators(self):
        return self.train_generator, self.val_generator, self.test_generator

    @abc.abstractmethod
    def create_file_list(self):
        pass

    def print_info(self):
        """
        Print the dataset information
        """

        print("="*30)
        print("Dataset name:        ", self.name)
        print("Dataset description: ", self.description)
        print("Input shape:         ", self.in_shape)
        print("Output shape:        ", self.out_shape)
        print("Number of subjects:  ", self.num_subjects)
        print("Number of samples:   ", self.num_files)
        print("="*30)


class bSSFPFineTuneDatasetLoader(bSSFPBaseDatasetLoader):
    def __init__(self,
                 data_path,
                 aug_fact,
                 train_test_split,
                 validate_test_split,
                 random_seed,
                 shard=0):
        super().__init__(data_path, aug_fact, train_test_split,
                         validate_test_split, random_seed)

        self.name = "bSSFP fine-tune"
        self.description = ("bSSFP fine-tune Dataset;"
                            " x are complex bSSFP 4D images,"
                            " y are DWI tensors")

    def create_file_list(self):
        """
        Get list of the files from the BIDS dataset.
        Split into training and testing sets.
        """
        self.subjects = self.bids_layout.get_subjects()
        self.num_subjects = len(self.subjects)

        all_complex = self.bids_layout.get(scope='preproc-dove',
                                           datatype='anat',
                                           suffix='bssfp',
                                           extension='nii.gz',
                                           return_type='filename')
        all_complex = [x for x in all_complex if 'complex' in x]
        y_fnames = []
        x_fnames = []
        for xfname in all_complex:
            entities = self.bids_layout.parse_file_entities(xfname)
            yfname = self.bids_layout.get(
                    scope='preproc-svenja',
                    subject=entities['subject'],
                    session=entities['session'],
                    datatype='dwi',
                    space='t1w',
                    desc='tensor',
                    suffix='dwi',
                    extension='nii.gz',
                    return_type='filename')

            if len(yfname) == 0:
                continue

            y_fnames.append(yfname[0])
            x_fnames.append(xfname)

        assert len(x_fnames) == len(y_fnames)

        example_input = nib.load(x_fnames[0])
        self.in_shape = (example_input.shape[:-1]
                         + (2 * example_input.shape[-1],))
        example_output = nib.load(y_fnames[0])
        self.out_shape = example_output.shape

        self.num_files = len(x_fnames)
        self.output_channels = self.out_shape[-1]
        self.input_channels = self.in_shape[-1]

        self.filenames = []
        for xfname, yfname in zip(x_fnames, y_fnames):
            self.filenames.append([xfname, yfname])

        n_train = int(self.num_subjects * self.train_test_split)
        n_val = int((self.num_subjects - n_train) * self.validate_test_split)
        self.train_subjects = self.subjects[:n_train]
        val_test_subjects = self.subjects[n_train:]
        self.val_subjects = val_test_subjects[n_val:]
        self.test_subjects = val_test_subjects[:n_val]

        self.train_filenames = [x for x in self.filenames
                                for sub in self.train_subjects
                                if sub in x[0]]
        self.val_filenames = [x for x in self.filenames
                              for sub in self.val_subjects
                              if sub in x[0]]
        self.test_filenames = [x for x in self.filenames
                               for sub in self.test_subjects
                               if sub in x[0]]


class bSSFPPretrainDatasetLoader(bSSFPBaseDatasetLoader):
    def __init__(self,
                 data_path,
                 aug_fact,
                 train_test_split,
                 validate_test_split,
                 random_seed,
                 shard=0):
        super().__init__(data_path, aug_fact, train_test_split,
                         validate_test_split, random_seed)

        self.name = "bSSFP Pre-Train"
        self.description = ("bSSFP pre-train Dataset;"
                            " x are complex bSSFP 4D images,"
                            " y are as x")

    def create_file_list(self):
        """
        Get list of the files from the BIDS dataset.
        Split into training and testing sets.
        """
        self.subjects = self.bids_layout.get_subjects()
        self.num_subjects = len(self.subjects)

        all_complex = self.bids_layout.get(scope='preproc-dove',
                                           datatype='anat',
                                           suffix='bssfp',
                                           extension='nii.gz',
                                           return_type='filename')
        all_complex = [x for x in all_complex if 'complex' in x]

        example_input = nib.load(all_complex[0])
        self.in_shape = (example_input.shape[:-1]
                         + (2 * example_input.shape[-1],))
        self.out_shape = self.in_shape

        self.num_files = len(all_complex)
        self.output_channels = self.out_shape[-1]
        self.input_channels = self.in_shape[-1]

        self.filenames = []
        for xfname, yfname in zip(all_complex, all_complex):
            self.filenames.append([xfname, yfname])

        n_train = int(self.num_subjects * self.train_test_split)
        n_val = int((self.num_subjects - n_train) * self.validate_test_split)
        self.train_subjects = self.subjects[:n_train]
        val_test_subjects = self.subjects[n_train:]
        self.val_subjects = val_test_subjects[n_val:]
        self.test_subjects = val_test_subjects[:n_val]

        self.train_filenames = [x for x in self.filenames
                                for sub in self.train_subjects
                                if sub in x[0]]
        self.val_filenames = [x for x in self.filenames
                              for sub in self.val_subjects
                              if sub in x[0]]
        self.test_filenames = [x for x in self.filenames
                               for sub in self.test_subjects
                               if sub in x[0]]


if __name__ == "__main__":
    data_loader = bSSFPFineTuneDatasetLoader(
            '/home/someusername/workspace/DOVE/bids',
            batch_size=4,
            train_test_split=0.8,
            validate_test_split=0.5,
            random_seed=42)
    data_loader.print_info()
    data_loader.ds_train, data_loader.ds_val, data_loader.ds_test = data_loader.get_dataset()
    data_loader.display_test_images()

#    ds = tf.data.Dataset.from_generator(data_loader,
#                                        output_types=(tf.float32, tf.float32),
#                                        output_shapes=(data_loader.in_shape,
#                                                       data_loader.out_shape))
#
#    ds = ds.batch(4)
