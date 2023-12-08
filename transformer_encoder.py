#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import *

def tf_enc(inputs, head_size, num_heads, ff_dim, dropout, LN):
    if LN=="B2T":#B2T connection
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
        x = Dropout(dropout)(x)
        res = x + inputs
        res = LayerNormalization(epsilon=1e-6)(res)
        # Feed Forward Part
        x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = Dropout(dropout)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = x + res + inputs
        x =  LayerNormalization(epsilon=1e-6)(x)
        return x

    if LN=="pre":# Normalization then Attention
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = Dropout(dropout)(x)
        res = x + inputs
        # Feed Forward Part
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        # x = GRU(ff_dim, return_sequences=True)(x)
        x = Dropout(dropout)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        # x = GRU(inputs.shape[-1], return_sequences=True)(x)
        return x + res

    if LN=="post":#Attention then Normalization
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
        x = Dropout(dropout)(x)
        res = x + inputs
        res = LayerNormalization(epsilon=1e-6)(res)
        # Feed Forward Part
        # x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        # x = Dropout(dropout)(x)
        # x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = GRU(ff_dim, return_sequences=True, activation="relu")(res)
        x = Dropout(dropout)(x)
        x = GRU(inputs.shape[-1], return_sequences=True, activation="relu")(x)

        x = x + res
        x =  LayerNormalization(epsilon=1e-6)(x)
        return x

    raise ValueError("LNが無効な変数です")

def build_model(
    input_,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    dropout,
    LN="pre",
    GAP=True
):
    x = input_
    for _ in range(num_transformer_blocks):
        x = tf_enc(x, head_size, num_heads, ff_dim, dropout, LN)

    if LN=="pre":
        x=LayerNormalization(epsilon=1e-6)(x)

    if GAP:
        x = GlobalAveragePooling1D(data_format="channels_first")(x)

    return x

