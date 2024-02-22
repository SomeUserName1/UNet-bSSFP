import abc

import tensorflow as tf
import numpy as np

import bids
import nibabel as nib
import pdb


# Adapted from IntelAI/unet
# https://github.com/IntelAI/unet/blob/master/3D/dataloader.py
class bSSFPBaseDatasetGenerator(abc.ABC):
    def __init__(self,
                 data_path,
                 batch_size,
                 train_test_split,
                 validate_test_split,
                 random_seed,
                 shard=0):
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.validate_test_split = validate_test_split
        self.random_seed = random_seed
        self.shard = shard  # For Horovod, gives different shard per worker

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

        self.ds_train, self.ds_val, self.ds_test = self.get_dataset()

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

    # TODO extend augmentation
    def augment_data(self, img, omap):
        """
        Data augmentation
        Flip image and mask. Rotate image and mask.
        """

        # Determine if axes are equal and can be rotated
        # If the axes aren't equal then we can't rotate them.
        equal_dim_axis = []
        for idx in range(0, len(self.shape)):
            for jdx in range(idx+1, len(self.shape)):
                if self.shape[idx] == self.shape[jdx]:
                    equal_dim_axis.append([idx, jdx])  # Valid rotation axes
        dim_to_rotate = equal_dim_axis

        if np.random.rand() > 0.5:
            # Random 0,1 (axes to flip)
            ax = np.random.choice(np.arange(len(self.shape)-1))
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

    def read_nifti_file(self, idx, augment=False):
        """
        Read Nifti file
        """
        idx = idx.numpy()
        imgFile = self.filenames[idx][0]
        omapFile = self.filenames[idx][1]

        img = np.array(nib.load(imgFile).dataobj)
        img = np.stack([np.abs(img), np.angle(img)], axis=-1)
        img = np.reshape(img, self.in_shape)

        omap = np.array(nib.load(omapFile).dataobj)
        omap.swapaxes(0, -1)

        # Normalize
        img = self.z_normalize_img(img)

        if augment:
            img, omap = self.augment_data(img, omap)

        return img, omap

    def plot_images(self, ds, slice_num=90):
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
                plt.imshow(omap[idx, slice_num, :, :], cmap="bone")
                plt.title("MRI", fontsize=18)

        plt.show()

    def display_train_images(self, slice_num=90):
        """
        Plots some training images
        """
        self.plot_images(self.ds_train, slice_num)

    def display_validation_images(self, slice_num=90):
        """
        Plots some validation images
        """
        self.plot_images(self.ds_val, slice_num)

    def display_test_images(self, slice_num=90):
        """
        Plots some test images
        """
        self.plot_images(self.ds_test, slice_num)

    def get_train(self):
        """
        Return train dataset
        """
        return self.ds_train

    def get_test(self):
        """
        Return test dataset
        """
        return self.ds_test

    def get_validate(self):
        """
        Return validation dataset
        """
        return self.ds_val

    def get_dataset(self):
        """
        Create a TensorFlow data loader
        """
        self.num_train = int(self.num_files * self.train_test_split)
        numValTest = self.num_files - self.num_train

        ds = tf.data.Dataset.range(self.num_files).shuffle(
                self.num_files, self.random_seed)  # Shuffle the dataset

        """
        Horovod Sharding
        Here we are not actually dividing the dataset into shards
        but instead just reshuffling the training dataset for every
        shard. Then in the training loop we just go through the training
        dataset but the number of steps is divided by the number of shards.
        """
        ds_train = ds.take(self.num_train).shuffle(
                self.num_train, self.shard)  # Reshuffle based on shard
        ds_val_test = ds.skip(self.num_train)
        self.num_val = int(numValTest * self.validate_test_split)
        self.num_test = self.num_train - self.num_val
        ds_val = ds_val_test.take(self.num_val)
        ds_test = ds_val_test.skip(self.num_val)

        ds_train = ds_train.map(
                lambda x: tf.py_function(self.read_nifti_file,
                                         [x, True],
                                         [tf.float32, tf.float32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
        ds_val = ds_val.map(
                lambda x: tf.py_function(self.read_nifti_file,
                                         [x, False],
                                         [tf.float32, tf.float32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
        ds_test = ds_test.map(
                lambda x: tf.py_function(self.read_nifti_file,
                                         [x, False],
                                         [tf.float32, tf.float32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds_train = ds_train.repeat()
        ds_train = ds_train.batch(self.batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        batch_size_val = 4
        ds_val = ds_val.batch(batch_size_val)
        ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

        batch_size_test = 1
        ds_test = ds_test.batch(batch_size_test)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        return ds_train, ds_val, ds_test


class bSSFPFineTuneDatasetGenerator(bSSFPBaseDatasetGenerator):
    def __init__(self,
                 data_path,
                 batch_size,
                 train_test_split,
                 validate_test_split,
                 random_seed,
                 shard=0):
        super().__init__(data_path, batch_size, train_test_split,
                         validate_test_split, random_seed, shard)

        self.name = "bSSFP fine-tune"
        self.description = ("bSSFP fine-tune Dataset;"
                            " x are complex bSSFP 4D images,"
                            " y are DWI tensors")

    def create_file_list(self):
        """
        Get list of the files from the BIDS dataset.
        Split into training and testing sets.
        """
        self.num_subjects = len(self.bids_layout.get_subjects())

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
                    space='t1w'
                    desc='tensor',
                    suffix='dwi',
                    extension='nii.gz',
                    return_type='filename')

            if len(yfname) == 0:
                continue

            y_fnames.append(yfname[0])
            x_fnames.append(xfname)

        pdb.set_trace()

        assert len(x_fnames) == len(y_fnames)

        example_input = nib.load(x_fnames[0])
        self.in_shape = (example_input.shape[:-1]
                         + (2 * example_input.shape[-1],))
        example_output = nib.load(y_fnames[0])
        self.out_shape = example_output.shape

        self.num_files = len(x_fnames)
        self.output_channels = self.out_shape[-1]
        self.input_channels = self.in_shape[-1]

        self.filenames = {}
        for idx, (xfname, yfname) in enumerate(zip(x_fnames, y_fnames)):
            self.filenames[idx] = [xfname, yfname]


if __name__ == "__main__":

    print("Load the data and plot a few examples")

    from argparser import args

    shape = (args.tile_height, args.tile_width,
             args.tile_depth, args.number_input_channels)

    """
    Load the dataset
    """
    brats_data = bSSFPFineTuneDatasetGenerator(
            data_path=args.data_path,
            batch_size=args.batch_size,
            train_test_split=args.train_test_split,
            validate_test_split=args.validate_test_split,
            number_output_classes=args.number_output_classes,
            random_seed=args.random_seed)

    brats_data.print_info()

