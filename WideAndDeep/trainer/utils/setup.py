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

import json
import logging
import os

import dllogger
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import tensorflow_transform as tft
from data.outbrain.dataloader import RawBinaryDataset, DatasetMetadata, data_input_fn
from data.outbrain.features import PREBATCH_SIZE


def init_cpu(args, logger):
    hvd.init()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    init_logger(
        full=hvd.rank() == 0,
        args=args,
        logger=logger
    )

    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(16)
    
    if args.amp:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if args.xla:
        tf.config.optimizer.set_jit(True)
    
    logger.warning('--gpu flag not set, running computation on CPU')


def init_logger(args, full, logger):
    if full:
        logger.setLevel(logging.INFO)
        log_path = os.path.join(args.results_dir, args.log_filename)
        os.makedirs(args.results_dir, exist_ok=True)
        dllogger.init(backends=[
            dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                       filename=log_path),
            dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)])
        logger.warning('command line arguments: {}'.format(json.dumps(vars(args))))
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)

        with open('{}/args.json'.format(args.results_dir), 'w') as f:
            json.dump(vars(args), f, indent=4)
    else:
        logger.setLevel(logging.ERROR)
        dllogger.init(backends=[])

    # dllogger.log(data=vars(args), step='PARAMETER')


def create_input_pipelines(dataset_path, train_batch_size, eval_batch_size, 
                            numerical_features, prefetch_batches):
    
    dataset_metadata = RawBinaryDataset.get_metadata(dataset_path, numerical_features)
    embeddings = range(0, 26)
    embedding_sizes = dataset_metadata.categorical_cardinalities
    train_dataset = RawBinaryDataset(data_path=dataset_path,
                             batch_size=train_batch_size,
                             numerical_features=numerical_features,
                             categorical_features=embeddings,
                             categorical_feature_sizes=embedding_sizes,
                             prefetch_depth=prefetch_batches,
                             drop_last_batch=True)

    test_dataset = RawBinaryDataset(data_path=dataset_path,
                            batch_size=eval_batch_size,
                            numerical_features=numerical_features,
                            categorical_features=embeddings,
                            categorical_feature_sizes=embedding_sizes,
                            prefetch_depth=prefetch_batches,
                            drop_last_batch=True)
    return train_dataset, test_dataset, dataset_metadata


def create_config(args):
    # assert not (args.cpu and args.amp), \
    #     'Automatic mixed precision conversion works only with GPU'
    assert not args.benchmark or args.benchmark_warmup_steps < args.benchmark_steps, \
        'Number of benchmark steps must be higher than warmup steps'
    logger = logging.getLogger('tensorflow')

    if args.cpu:
        init_cpu(args, logger)

    num_gpus = hvd.size()
    gpu_id = hvd.rank()
    train_batch_size = args.global_batch_size // num_gpus
    eval_batch_size = args.eval_batch_size // num_gpus
    steps_per_epoch = args.training_set_size / args.global_batch_size
    eval_point = args.eval_point

    feature_spec = tft.TFTransformOutput(
        '/root'
    ).transformed_feature_spec()
    # train_dataset, test_dataset, dataset_metadata = create_input_pipelines(args.train_dataset_path, train_batch_size, eval_batch_size, 13, 10)
    train_dataset = data_input_fn(
        args.train_data_pattern,
        feature_spec,
        train_batch_size // PREBATCH_SIZE,
        num_gpus,
        gpu_id)
    test_dataset = data_input_fn(
        args.eval_data_pattern, 
        feature_spec,
        eval_batch_size // PREBATCH_SIZE,
        num_gpus,
        gpu_id)

    # steps_per_epoch = train_dataset.cardinality()
    # print(f'steps: {steps_per_epoch}')
    config = {
        'steps_per_epoch': steps_per_epoch,
        'train_dataset': train_dataset,
        'eval_dataset': test_dataset, 
        'eval_point': eval_point
    }

    return config
