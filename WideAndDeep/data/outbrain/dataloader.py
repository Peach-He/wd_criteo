# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import tensorflow as tf

import concurrent
import math
import os
import queue
import json
from collections import namedtuple

import numpy as np
from typing import Optional, Sequence, Tuple, Any, Dict

from functools import partial
import struct
from multiprocessing import cpu_count


DatasetMetadata = namedtuple('DatasetMetadata', ['num_numerical_features',
                                                 'categorical_cardinalities'])


def get_categorical_feature_type(size: int):
    types = (np.int8, np.int16, np.int32)

    for numpy_type in types:
        if size < np.iinfo(numpy_type).max:
            return numpy_type

    raise RuntimeError(f"Categorical feature of size {size} is too big for defined types")


class RawBinaryDataset:
    """Split version of Criteo dataset

    Args:
        data_path (str): Full path to split binary file of dataset. It must contain numerical.bin, label.bin and
            cat_0 ~ cat_25.bin
        batch_size (int):
        numerical_features(boolean): Number of numerical features to load, default=0 (don't load any)
        categorical_features (list or None): categorical features used by the rank (IDs of the features)
        categorical_feature_sizes (list of integers): max value of each of the categorical features
        prefetch_depth (int): How many samples to prefetch. Default 10.
    """

    _model_size_filename = 'model_size.json'

    def __init__(
        self,
        data_path: str,
        batch_size: int = 1,
        numerical_features: int = 0,
        categorical_features: Optional[Sequence[int]] = None,
        categorical_feature_sizes: Optional[Sequence[int]] = None,
        prefetch_depth: int = 10,
        drop_last_batch: bool = False
    ):
        self.tar_fea = 1   # single target
        self.den_fea = 13  # 13 dense  features
        self.spa_fea = 26  # 26 sparse features
        self.tad_fea = self.tar_fea + self.den_fea
        self.tot_fea = self.tad_fea + self.spa_fea
        INT_COLS = list(range(1, 14))
        CAT_COLS = list(range(14, 40))

        self.NUMERIC_COLUMNS = ['c%d' % i for i in INT_COLS]

        self.CATEGORICAL_COLUMNS = ['c%d' % i for i in CAT_COLS]

        data_path = os.path.join(data_path, 'train')
        self._label_bytes_per_batch = np.dtype(np.bool).itemsize * batch_size
        self._numerical_bytes_per_batch = numerical_features * np.dtype(np.float16).itemsize * batch_size
        self._single_num_bytes_per_batch = np.dtype(np.float16).itemsize * batch_size
        self._numerical_features = numerical_features

        self._categorical_feature_types = [
            get_categorical_feature_type(size) for size in categorical_feature_sizes
        ] if categorical_feature_sizes else []
        self._categorical_bytes_per_batch = [
            np.dtype(cat_type).itemsize * batch_size for cat_type in self._categorical_feature_types
        ]

        self._categorical_features = categorical_features
        self._batch_size = batch_size

        self._label_file = os.open(os.path.join(data_path, 'label.bin'), os.O_RDONLY)
        self._num_entries = int(math.ceil(os.fstat(self._label_file).st_size
                                          / self._label_bytes_per_batch)) if not drop_last_batch \
                            else int(math.floor(os.fstat(self._label_file).st_size / self._label_bytes_per_batch))
        self._numerical_features_file = os.open(os.path.join(data_path, "numerical.bin"), os.O_RDONLY)
        self._categorical_features_files = []
        for cat_id in categorical_features:
            cat_file = os.open(os.path.join(data_path, f"cat_{cat_id}.bin"), os.O_RDONLY)
            cat_bytes = self._categorical_bytes_per_batch[cat_id]
            number_of_categorical_batches = math.ceil(os.fstat(cat_file).st_size / cat_bytes) if not drop_last_batch \
                                            else math.floor(os.fstat(cat_file).st_size / cat_bytes)
            if number_of_categorical_batches != self._num_entries:
                raise ValueError(f"Size mismatch in data files. Expected: {self._num_entries}, got: {number_of_categorical_batches}")
            self._categorical_features_files.append(cat_file)
        

        self._prefetch_depth = min(prefetch_depth, self._num_entries)
        self._prefetch_queue = queue.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    @classmethod
    def get_metadata(cls, path, num_numerical_features):
        with open(os.path.join(path, cls._model_size_filename), 'r') as f:
            global_table_sizes = json.load(f)

        global_table_sizes = list(global_table_sizes.values())
        global_table_sizes = [s + 1 for s in global_table_sizes]

        metadata = DatasetMetadata(num_numerical_features=num_numerical_features,
                                   categorical_cardinalities=global_table_sizes)
        return metadata

    def __len__(self):
        return self._num_entries

    def __getitem__(self, idx: int):
        if idx >= self._num_entries:
            raise IndexError()

        if self._prefetch_depth <= 1:
            return self._get_item(idx)

        if idx == 0:
            for i in range(self._prefetch_depth):
                self._prefetch_queue.put(self._executor.submit(self._get_item, (i)))
        if idx < self._num_entries - self._prefetch_depth:
            self._prefetch_queue.put(self._executor.submit(self._get_item, (idx + self._prefetch_depth)))
        return self._prefetch_queue.get().result()
        # return self._get_item(idx)

    def _get_item(self, idx: int) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        click = self._get_label(idx)
        numerical_features = self._get_numerical_features(idx)
        categorical_features = self._get_categorical_features(idx)
        # features = tf.concat([numerical_features, categorical_features], axis=0)
        return {**numerical_features, **categorical_features}, click

    def _get_label(self, idx: int) -> tf.Tensor:
        raw_label_data = os.pread(self._label_file, self._label_bytes_per_batch,
                                  idx * self._label_bytes_per_batch)
        array = np.frombuffer(raw_label_data, dtype=np.bool)
        array = tf.convert_to_tensor(array, dtype=tf.float32)
        array = tf.expand_dims(array, 1)
        return array

    def _get_numerical_features(self, idx: int) -> Optional[tf.Tensor]:
        if self._numerical_features_file is None:
            return -1

        # raw_numerical_data = os.pread(self._numerical_features_file, self._numerical_bytes_per_batch,
        #                               idx * self._numerical_bytes_per_batch)
        # array = np.frombuffer(raw_numerical_data, dtype=np.float16)
        # array = tf.convert_to_tensor(array, name='dense_input')
        # return tf.reshape(array, shape=[self._batch_size, self._numerical_features])
        numerical_features = {}
        for col in self.NUMERIC_COLUMNS:
            raw_num_data = os.pread(self._numerical_features_file, self._single_num_bytes_per_batch, idx * self._single_num_bytes_per_batch)
            array = np.frombuffer(raw_num_data, dtype=np.float16)
            tensor = tf.convert_to_tensor(array)
            tensor = tf.expand_dims(tensor, axis=1)
            # numerical_features.append(tensor)
            numerical_features[col] = tensor
        return numerical_features

    def _get_categorical_features(self, idx: int) -> Optional[tf.Tensor]:
        if self._categorical_features_files is None:
            return -1

        categorical_features = {}
        for cat_id, cat_file in zip(self._categorical_features, self._categorical_features_files):
            cat_bytes = self._categorical_bytes_per_batch[cat_id]
            cat_type = self._categorical_feature_types[cat_id]
            raw_cat_data = os.pread(cat_file, cat_bytes, idx * cat_bytes)
            array = np.frombuffer(raw_cat_data, dtype=cat_type)
            tensor = tf.convert_to_tensor(array)
            tensor = tf.expand_dims(tensor, axis=1)
            # categorical_features.append(tensor)
            categorical_features[f'c{cat_id + 14}'] = tensor
        return categorical_features

    # def __del__(self):
    #     if self._file is not None:
    #         os.close(self._file)

INT_COLS = list(range(1, 14))
CAT_COLS = list(range(14, 40))
LABEL_COLUMN = 'c0'
NUMERIC_COLUMNS = ['c%d' % i for i in INT_COLS]
CATEGORICAL_COLUMNS = ['c%d' % i for i in CAT_COLS]
FEATURES = [LABEL_COLUMN] + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def _consolidate_batch(elem):
    label = elem.pop(LABEL_COLUMN)
    reshaped_label = tf.reshape(label, [-1, label.shape[-1]])
    features = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

    reshaped_elem = {
        key: tf.reshape(elem[key], [-1, elem[key].shape[-1]])
        for key in elem
        if key in features
    }

    return reshaped_elem, reshaped_label


def data_input_fn(
    file_path,
    feature_spec,
    batch_size, 
    num_gpus=1,
    id=0):
    
    # bytes_per_feature = np.__dict__['int32']().nbytes
    # record_width = 40
    # bytes_per_record = record_width * bytes_per_feature

    # feature_spec = gen_feature_spec()
    dataset = tf.data.Dataset.list_files(file_pattern=file_path, shuffle=None)

    dataset = tf.data.TFRecordDataset(
        filenames=dataset,
        num_parallel_reads=1
    )

    dataset = dataset.shard(num_gpus, id)

    dataset = dataset.shuffle(batch_size)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    dataset = dataset.apply(
        transformation_func=tf.data.experimental.parse_example_dataset(
            features=feature_spec,
            num_parallel_calls=1)
    )

    dataset = dataset.map(
        map_func=partial(_consolidate_batch),
        num_parallel_calls=None
    )

    dataset = dataset.prefetch(buffer_size=1)

    return dataset
