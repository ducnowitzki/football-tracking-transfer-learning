import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class SamplingTimestep(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        timesteps = tf.shape(z_mean)[1]
        dim = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, timesteps, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def get_feature_extractor(feature_dim, pitch_x_axis, pitch_y_axis, channels):
    inp = layers.Input((pitch_x_axis, pitch_y_axis, channels))
    x = layers.Conv2D(64, (5,5), strides=(2,2), padding='same')(inp)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (5,5), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    out = layers.Dense(units=feature_dim)(x)
    return keras.Model(inp, out, name="frame_feature_extractor")

def get_reverse_feature_extractor(feature_dim, pitch_x_axis, pitch_y_axis, channels):
    z_inp = keras.Input((feature_dim,))
    x = layers.Dense(units=128*int((1/4)*pitch_x_axis)*(1/4)*pitch_y_axis)(z_inp)
    x = layers.Reshape(target_shape=(int((1/4)*pitch_x_axis),int((1/4)*pitch_y_axis),128))(x)
    x = layers.Conv2DTranspose(128, (5,5), padding="same", strides=(1,1), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (5,5), padding="same", strides=(2,2), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    reconstruction = layers.Conv2DTranspose(channels, (5,5), strides=(2,2), padding="same", activation="tanh")(x)
    reconstruction = layers.Conv2DTranspose(channels, (5,5), strides=(1,1), padding="same", activation="sigmoid")(reconstruction)
    if pitch_x_axis % 2 != 0:
        reconstruction = layers.ZeroPadding2D(padding=((1,0),(0,0)))(reconstruction)
    return keras.Model(z_inp, reconstruction, name="reconstruction_decoder")

def get_recurrent_encoder(hidden_dim, intermediate_dim, pitch_x_axis, pitch_y_axis, channels, timesteps, feature_extractor):
    game_sequence = layers.Input(shape=(timesteps, pitch_x_axis, pitch_y_axis, channels))
    feature_sequence = layers.TimeDistributed(feature_extractor)(game_sequence)
    h = layers.LSTM(intermediate_dim, return_sequences=True)(feature_sequence)
    return keras.Model(game_sequence, h, name="encoder")

def get_timewise_MLP_a(timesteps, intermediate_dim, hidden_dim):
    inp = layers.Input((timesteps, intermediate_dim))
    z_mean_t = layers.TimeDistributed(layers.Dense(hidden_dim))(inp)
    z_log_sigma_t = layers.TimeDistributed(layers.Dense(hidden_dim))(inp)
    z_t = SamplingTimestep()([z_mean_t, z_log_sigma_t])
    return keras.Model(inp, [z_mean_t, z_log_sigma_t, z_t])

def get_timewise_MLP_z(timesteps, intermediate_dim, hidden_dim):
    feature_inp = layers.Input((timesteps, intermediate_dim))
    hidden_inp = layers.Input((timesteps, hidden_dim))
    inp = layers.Concatenate()([feature_inp, hidden_inp])
    z_mean_t = layers.TimeDistributed(layers.Dense(hidden_dim))(inp)
    z_log_sigma_t = layers.TimeDistributed(layers.Dense(hidden_dim))(inp)
    z_t = SamplingTimestep()([z_mean_t, z_log_sigma_t])
    return keras.Model([feature_inp, hidden_inp], [z_mean_t, z_log_sigma_t, z_t])

def get_recurrent_decoder(timesteps, hidden_dim, intermediate_dim, feature_dim, reverse_feature_extractor):
    inp_a = layers.Input((timesteps, hidden_dim))
    inp_z = layers.Input((timesteps, hidden_dim))
    inp = layers.Concatenate()([inp_a, inp_z])
    decoder_h = layers.LSTM(intermediate_dim, return_sequences=True)(inp)
    decoder_mean = layers.LSTM(feature_dim, return_sequences=True)(decoder_h)
    x_decoded_mean_sequence = layers.TimeDistributed(reverse_feature_extractor)(decoder_mean)
    return keras.Model([inp_a, inp_z], x_decoded_mean_sequence)

def get_classifier(timesteps, hidden_dim, no_classes):
    a = layers.Input((timesteps, hidden_dim))
    pred = layers.TimeDistributed(layers.Dense(no_classes, activation='softmax'))(a)
    return keras.Model(a, pred)

class SeqLabelVAE(keras.Model):
    def __init__(self, feature_dim, intermediate_dim, hidden_dim, pitch_x_axis, pitch_y_axis, 
                 channels, timesteps, no_classes, **kwargs):
        super(SeqLabelVAE, self).__init__(**kwargs)
        
        # Build components
        self.feature_extractor = get_feature_extractor(feature_dim, pitch_x_axis, pitch_y_axis, channels)
        self.reverse_feature_extractor = get_reverse_feature_extractor(feature_dim, pitch_x_axis, pitch_y_axis, channels)
        self.recurrent_encoder = get_recurrent_encoder(hidden_dim, intermediate_dim, pitch_x_axis, 
                                                     pitch_y_axis, channels, timesteps, self.feature_extractor)
        self.MLP_a = get_timewise_MLP_a(timesteps, intermediate_dim, hidden_dim)
        self.MLP_z = get_timewise_MLP_z(timesteps, intermediate_dim, hidden_dim)
        self.recurrent_decoder = get_recurrent_decoder(timesteps, hidden_dim, intermediate_dim, 
                                                     feature_dim, self.reverse_feature_extractor)
        self.classifier = get_classifier(timesteps, hidden_dim, no_classes)
        
    def encode(self, x):
        h_t = self.recurrent_encoder(x)
        a_mean_t, a_log_sigma_t, a_t = self.MLP_a(h_t)
        z_mean_t, z_log_sigma_t, z_t = self.MLP_z([h_t, a_t])
        return a_mean_t, a_log_sigma_t, a_t, z_mean_t, z_log_sigma_t, z_t
    
    def decode(self, a_t, z_t):
        return self.recurrent_decoder([a_t, z_t])
    
    def classify(self, a_t):
        return self.classifier(a_t)
    
    def call(self, inputs):
        a_mean_t, a_log_sigma_t, a_t, z_mean_t, z_log_sigma_t, z_t = self.encode(inputs)
        reconstruction = self.decode(a_t, z_t)
        classification = self.classify(a_t)
        return reconstruction, classification, a_mean_t, a_log_sigma_t, z_mean_t, z_log_sigma_t 