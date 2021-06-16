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

import tensorflow as tf

from data.outbrain.features import get_feature_columns, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, HASH_BUCKET_SIZES, HASH_BUCKET_SIZE


def wide_deep_model(args):
    wide_columns, deep_columns = get_feature_columns()

    wide_weighted_outputs = []
    numeric_dense_inputs = []
    wide_columns_dict = {}
    deep_columns_dict = {}
    features = {}

    for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:
        features[col] = tf.keras.Input(shape=(1,),
                                           batch_size=None,
                                           name=col,
                                           dtype=tf.float32 if col in NUMERIC_COLUMNS else tf.int32,
                                           sparse=False)

    # for key in wide_columns_dict:
    #     if key in CATEGORICAL_COLUMNS:
    #         wide_weighted_outputs.append(tf.keras.layers.Flatten()(tf.keras.layers.Embedding(
    #             HASH_BUCKET_SIZES[key], 1, input_length=1)(features[key])))
    #     else:
    #         numeric_dense_inputs.append(features[key])

    # categorical_output_contrib = tf.keras.layers.add(wide_weighted_outputs,
    #                                                  name='categorical_output')
    # numeric_dense_tensor = tf.keras.layers.concatenate(
    #     numeric_dense_inputs, name='numeric_dense')

    dnn = tf.keras.layers.DenseFeatures(feature_columns=deep_columns)(features)
    for unit_size in args.deep_hidden_units:
        dnn = tf.keras.layers.Dense(units=unit_size, activation='relu')(dnn)
        dnn = tf.keras.layers.Dropout(rate=args.deep_dropout)(dnn)
        dnn = tf.keras.layers.BatchNormalization()(dnn)
    dnn = tf.keras.layers.Dense(units=1)(dnn)
    dnn_model = tf.keras.Model(inputs=features,
                               outputs=dnn)

    linear = tf.keras.layers.DenseFeatures(wide_columns)(features)
    linear_output = tf.keras.layers.Dense(1)(linear)

    linear_model = tf.keras.Model(inputs=features,
                                  outputs=linear_output)

    model = tf.keras.experimental.WideDeepModel(
        linear_model, dnn_model, activation='sigmoid')

    return model


def wide_deep_model_mod(args):
    dense_inputs = tf.keras.Input(shape=(13,), dtype=tf.float32, name='dense_input')
    sparse_inputs = tf.keras.Input(shape=(26,), dtype=tf.int32, name='sparse_input')
    
    embedded = tf.keras.layers.Embedding(HASH_BUCKET_SIZES['c14'], 128)(sparse_inputs[:, 0])
    cnt = 0
    for i in HASH_BUCKET_SIZES:
        embedded_tensor = tf.keras.layers.Embedding(HASH_BUCKET_SIZES[i], 128)(sparse_inputs[:, cnt])
        embeded = tf.keras.layers.Concatenate()([embedded, embedded_tensor])
        cnt += 1

    # dnn_embedded = tf.keras.layers.Embedding(100000000, 128)(sparse_inputs)
    # linear_embedded = tf.keras.layers.Embedding(100000000, 1)(sparse_inputs)

    # dnn_embedded_concat = tf.keras.layers.concatenate([dnn_embedded[:, 0, :], dnn_embedded[:, 1, :]])
    # for i in range(26):
    #     dnn_embedded_concat = tf.keras.layers.concatenate([dnn_embedded_concat, dnn_embedded[:, i, :]])

    # tf.print(dnn_embedded.shape)
    # tf.print(linear_embedded.shape)
    # tf.print(dnn_embedded_concat.shape)

    # dnn = tf.keras.layers.concatenate([dense_inputs, dnn_embedded_concat], name='concat_dnn')
    # linear = tf.keras.layers.concatenate([dense_inputs, linear_embedded[:, 0, :]], name='linear_concat')
    dnn = tf.keras.layers.Concatenate()([dense_inputs, embedded])
    linear = dnn
    for unit_size in args.deep_hidden_units:
        dnn = tf.keras.layers.Dense(units=unit_size, activation='relu')(dnn)
        dnn = tf.keras.layers.Dropout(rate=args.deep_dropout)(dnn)
        dnn = tf.keras.layers.BatchNormalization()(dnn)
    dnn = tf.keras.layers.Dense(units=1)(dnn)
    dnn_model = tf.keras.Model(inputs=[dense_inputs, sparse_inputs],
                               outputs=dnn)
    linear_output = tf.keras.layers.Dense(1)(linear)

    linear_model = tf.keras.Model(inputs=[dense_inputs, sparse_inputs],
                                  outputs=linear_output)

    logit = tf.keras.layers.Add()([dnn, linear_output])
    output = tf.sigmoid(logit)
    # model = tf.keras.experimental.WideDeepModel(
    #     dnn_model, dnn_model, activation='sigmoid')
    model = tf.keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=output)
    return model