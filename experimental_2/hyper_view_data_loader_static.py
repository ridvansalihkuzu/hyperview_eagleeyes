
import tensorflow as tf
import pandas as pd
from functools import partial
import albumentations as A
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split



class DataGenerator():
    def __init__(self,train_dir, label_dir, eval_dir,valid_size=0.2,image_shape=(128,128), batch_size=16):
        """Constructor.
                """
        self.train_dir=train_dir
        self.label_dir=label_dir
        self.eval_dir=eval_dir
        self.batch_size = batch_size

        train_stats_log = '{}/stats.npy'.format(train_dir)
        eval_stats_log = '{}/stats.npy'.format(eval_dir)

        if (not os.path.exists(train_stats_log)):
            train_stats = DataGenerator._get_stats(train_dir)
            np.save(train_stats_log, train_stats)
        if (not os.path.exists(eval_stats_log)):
            eval_stats = DataGenerator._get_stats(eval_dir)
            np.save(eval_stats_log, eval_stats)

        self.train_stats = np.load(train_stats_log)
        self.eval_stats = np.load(eval_stats_log)
        tr_trans, val_trans, eval_trans = DataGenerator._init_transform(image_shape, self.train_stats,self.eval_stats)

        train_files = DataGenerator._load_data(train_dir)
        train_labels = DataGenerator._load_gt(label_dir)
        #train_labels = np.vsplit(train_labels, len(train_labels))
        train_files, valid_files, train_labels, valid_labels = train_test_split(train_files, train_labels, test_size = valid_size, random_state = 42)
        valid_files, test_files,  valid_labels, test_labels = train_test_split(valid_files, valid_labels,test_size=0.20, random_state=42)

        eval_files = DataGenerator._load_data(eval_dir)
        eval_labels=np.zeros(eval_files.shape)

        self.train_reader = DataGenerator._get_data_reader(train_files,train_labels,batch_size,tr_trans,image_shape,ext_aug=True,stats=self.train_stats)
        self.valid_reader = DataGenerator._get_data_reader(valid_files, valid_labels,batch_size, val_trans,image_shape,ext_aug=True,stats=self.train_stats)
        self.evalid_reader = DataGenerator._get_data_reader(valid_files, valid_labels, batch_size, eval_trans,image_shape,ext_aug=False, stats=self.train_stats)
        self.test_reader = DataGenerator._get_data_reader(test_files, test_labels, batch_size, eval_trans,image_shape,ext_aug=False,stats=self.eval_stats)
        self.eval_reader = DataGenerator._get_data_reader(eval_files, eval_labels,batch_size, eval_trans,image_shape,ext_aug=False,eval=True,stats=self.eval_stats)

        self.image_shape, self.label_shape = DataGenerator._get_dataset_features(self.valid_reader)


    @staticmethod
    def _get_dataset_features(reader):
        for feature, mask, in reader.take(1):
            image_shape=tuple([1, feature.shape[-3],feature.shape[-2],feature.shape[-1]])
            label_shape=mask.shape[-1]
            return image_shape,label_shape,

    @staticmethod
    def _get_data_reader(files, labels, batch_size, transform, image_shape, ext_aug=True, eval=False,stats=None):


        dataset = tf.data.Dataset.from_tensor_slices((files,labels))
        dataset = dataset.interleave(lambda x,y: DataGenerator._deparse_single_image(x, y,image_shape,ext_aug),cycle_length=batch_size,num_parallel_calls=tf.data.AUTOTUNE)
        if not eval:
            dataset = dataset.shuffle(buffer_size=int(len(files)/4), reshuffle_each_iteration=True)
        dataset = dataset.map(partial(DataGenerator._trans_single_image, transform=transform,eval=eval,stats=stats),num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=False,num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        #if batch_size<2:
        return dataset
        #else:
        #    options = tf.data.Options()
        #    options.experimental_distribute.auto_shard_policy = tf.data.experimental_1.AutoShardPolicy.DATA
        #    return dataset.with_options(options)

    @staticmethod
    def _load_gt(file_path: str):
        """Load labels for train set from the ground truth file.
        Args:
            file_path (str): Path to the ground truth .csv file.
        Returns:
            [type]: 2D numpy array with soil properties levels
        """
        gt_file = pd.read_csv(file_path)
        labels = gt_file[["P", "K", "Mg", "pH"]].values / np.array([325,625,400,7.8])
        return labels

    @staticmethod
    def _load_data(directory: str):
        all_files = np.array(
            sorted(
                glob(os.path.join(directory, "*.npz")),
                key=lambda x: int(os.path.basename(x).replace(".npz", "")),
            )
        )
        return all_files

    @staticmethod
    def preprocess_extract_patch(target_shape):
        def _preprocess_extract_patch(image):
            print(image.shape)
            max_edge = np.max(image.shape[1:])
            if max_edge < target_shape[0]:
                max_edge = target_shape
            else:
                max_edge = (max_edge, max_edge)
            image = DataGenerator._shape_pad(image, max_edge)
            image = image.transpose((1, 2, 0))
            return image

        return _preprocess_extract_patch

    @staticmethod
    def _shape_pad(data,shape):
        padded=np.pad(data,
                      ((0, 0),
                       (0, (shape[0] - data.shape[1])),
                       (0, (shape[1] - data.shape[2]))),
                      'wrap')
        #print(padded.shape)
        return padded

    @staticmethod
    def _deparse_single_image(filename, label, target_shape,ext_aug=True):
        import random
        def _read_npz(filename):
            with np.load(filename.numpy()) as npz:
                image = npz['data']
                mask = (1 - npz['mask'].astype(int))
                if ext_aug:
                    if random.choice([True, False]):
                        image = (image * mask)
                else:
                    image = (image * mask)

                sh=image.shape[1:]
                max_edge = np.max(sh)
                min_edge = np.min(sh) #AUGMENT BY SHAPE
                flag=True
                if False: #ext_aug:
                    if min_edge>32:
                        x = np.random.randint(sh[0]+1 - min_edge)
                        y = np.random.randint(sh[1]+1 - min_edge)
                        image=image[:,x:(x+min_edge),y:(y+min_edge)]
                        flag=False

                if flag:
                    if max_edge<target_shape[0]:
                        max_edge=target_shape
                    else:
                        max_edge=(max_edge,max_edge)
                    image = DataGenerator._shape_pad(image, max_edge)
                image = image.transpose((1, 2, 0))

                return image
        [image ] = tf.py_function(_read_npz, [filename], [tf.float32])

        return tf.data.Dataset.from_tensors((image, label,filename))

    @staticmethod
    def _trans_single_image(feature,label,filename,transform=None,eval=True,stats=None):
        def _aug_fn(image):

            image = image / np.max(stats[-1])  # MAX
            augmented = transform(image=image)
            feature = augmented['image']
            #feature=feature.transpose((2, 0, 1))
            #feature = np.nan_to_num(feature, nan=np.finfo(float).eps, posinf=np.finfo(float).eps, neginf=-np.finfo(float).eps)
            feature = tf.cast(feature, tf.float32)

            return feature
        if transform is not None:
            feature = tf.numpy_function(func=_aug_fn, inp=[feature], Tout=[tf.float32])
        if not eval:
            return tf.cast(feature, tf.float32), tf.cast(label, tf.float32)
        else:
            return tf.cast(feature, tf.float32), tf.cast(label, tf.float32),filename

    @staticmethod
    def _get_stats(directory: str):
        all_files = np.array(
            sorted(
                glob(os.path.join(directory, "*.npz")),
                key=lambda x: int(os.path.basename(x).replace(".npz", "")),
            )
        )
        data=[]
        for file_name in all_files:
            with np.load(file_name) as npz:
                arr = np.ma.MaskedArray(**npz)
                d = np.max(arr, (1, 2))
                data.append(d)
        data=np.array(data)
        max_data=np.max(data,0)

        fst_moment = 0
        snd_moment = 0
        total_pixel_sum = 0.0
        total_pixel_count = 0.0
        total_pixel_sum_sq = 0.0


        for file_name in all_files:
            with np.load(file_name) as npz:
                br=np.tile(max_data[:, np.newaxis, np.newaxis], npz['data'].shape[1:])
                arr = np.ma.MaskedArray(**npz)
                arr=arr/br
                #d = np.max(arr, (1, 2))
                #data.append(d)
                nb_pixels=np.sum(1 - npz['mask'][0].astype(int)).astype(np.float)
                total_pixel_sum = np.sum(arr.astype(np.float), (1, 2))
                total_pixel_sum_sq = np.sum(arr.astype(np.float) ** 2, (1, 2))
                fst_moment = (total_pixel_count * fst_moment + total_pixel_sum) / (total_pixel_count + nb_pixels)
                snd_moment = (total_pixel_count * snd_moment + total_pixel_sum_sq) / (total_pixel_count + nb_pixels)
                total_pixel_count += nb_pixels

                #https://www.binarystudy.com/2021/04/how-to-calculate-mean-standard-deviation-images-pytorch.html
        mean, std = fst_moment, np.sqrt(snd_moment - fst_moment ** 2)

        #total_mean = total_pixel_sum / total_pixel_count
        #total_var = (total_pixel_sum_sq / total_pixel_count) - (total_mean ** 2)
        #total_std = np.sqrt(total_var)
        return np.array([mean, std, max_data])

    @staticmethod
    def _init_transform(image_shape, train_stats, eval_stats):
        train_transform = A.Compose([
            A.Resize(image_shape[0], image_shape[1]),
            #A.Normalize(mean=train_stats[0]*train_stats[2], std=train_stats[1]*train_stats[2], max_pixel_value=1),
            A.GaussNoise(var_limit=0.000025,p=0.5),
            A.RandomRotate90(p=0.5),
            #A.Rotate(),
            A.RandomResizedCrop(image_shape[0], image_shape[1], ratio=(0.95, 1.05), p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=90, shift_limit_x=0.05, shift_limit_y=0.05,p=0.5),
            # A.RandomBrightnessContrast(),

        ])

        valid_transform = A.Compose([
            A.Resize(image_shape[0], image_shape[1]),
            # A.Normalize(mean=train_stats[0]*train_stats[2], std=train_stats[1]*train_stats[2], max_pixel_value=1),
            A.GaussNoise(var_limit=0.000025,p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomResizedCrop(image_shape[0], image_shape[1], ratio=(0.95, 1.05), p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=90, shift_limit_x=0.05, shift_limit_y=0.05,p=0.5),

        ])

        eval_transform = A.Compose([
            A.Resize(image_shape[0], image_shape[1]),
            #A.Normalize(mean=eval_stats[0]*eval_stats[2], std=eval_stats[1]*eval_stats[2], max_pixel_value=1)
            ])

        return train_transform, valid_transform, eval_transform




