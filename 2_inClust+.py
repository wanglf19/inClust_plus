#! -*- coding: utf-8 -*-
#the VAE and its variants (cVAE and VAE with clustering)could be refer to https://github.com/bojone/vae
import csv
from scipy import sparse
from keras import optimizers
import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K
from sklearn.decomposition import PCA
# import imageio,os
# from keras.datasets import mnist
# from keras.datasets import fashion_mnist as mnist
# from cortex import CortexDataset  #wang
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import os, csv, re
import argparse

#'''
#####################################################################################################################
parser = argparse.ArgumentParser(description='inClust+')
# system config
parser.add_argument('--inputdata', type=str, default='data/training_data/Fig1_merfish_data.npz', help='address for input data')
parser.add_argument('--input_covariates', type=str, default='data/training_data/Fig1_merfish_batch.npy', help='address for covariate (e.g. batch)')
parser.add_argument('--inputcelltype', type=str, default='data/training_data/Fig1_merfish_cell_type.npy', help='address for celltype label')
parser.add_argument('--permute_input', type=str, default='F', help='whether permute the input')
parser.add_argument('--randoms', type=int, default=30, help='random number to split dataset')

#+ new
parser.add_argument('--input_mask', type=str, default='data/training_data/Fig1_merfish_input_mask.npy', help='address for celltype label')
parser.add_argument('--output_mask', type=str, default='data/training_data/Fig1_merfish_output_mask.npy', help='address for celltype label')

parser.add_argument('--dim_latent', type=int, default=50, help='dimension of latent space')
parser.add_argument('--dim_intermediate', type=int, default=200, help='dimension of intermediate layer')
parser.add_argument('--activation', type=str, default='relu', help='activation function: relu or tanh')
parser.add_argument('--last_activation', type=str, default='relu', help='activation function: relu, tanh or sigmoid')
parser.add_argument('--arithmetic', type=str, default='minus', help='arithmetic: minus or plus')
parser.add_argument('--independent_embed', type=str, default='T', help='embedding mode')

parser.add_argument('--batch_size', type=int, default=500, help='training parameters_batch_size')
parser.add_argument('--epochs', type=int, default=50, help='training parameters_epochs')

parser.add_argument('--training', type=str, default='T', help='training model(T) or loading model(F) ')
parser.add_argument('--weights', type=str, default='data/weights_and_results/Fig2_demo.weight', help='trained weights')

parser.add_argument('--mode', type=str, default='suprevised', help='mode: supervised, semi_supervised, unsupervised, user_defined')
parser.add_argument('--task', type=str, default='integration', help='mode for inClust+:integration, imputation, generation')


parser.add_argument('--reconstruction_loss', type=int, default=5, help='The reconstruction loss for VAE')
parser.add_argument('--kl_cross_loss', type=int, default=1, help='')
parser.add_argument('--prior_distribution_loss', type=int, default=1, help='The assumption that prior distribution is uniform distribution')
parser.add_argument('--label_cross_loss', type=int, default=50, help='Loss for integrating label information into the model')



args = parser.parse_args()
inputdata = args.inputdata
inputbatch = args.input_covariates
inputcelltype = args.inputcelltype
randoms = args.randoms
permute_input = args.permute_input
arithmetic = args.arithmetic
independent_embed = args.independent_embed

input_mask_data = args.input_mask
output_mask_data = args.output_mask

z_dim = args.dim_latent
intermediate_dim = args.dim_intermediate
activation_function = args.activation
last_activation = args.last_activation

epochs = args.epochs
batch_size = args.batch_size

training = args.training
weights = args.weights

reconstruction_loss = args.reconstruction_loss
kl_cross_loss = args.kl_cross_loss
prior_distribution_loss = args.prior_distribution_loss
label_cross_loss = args.label_cross_loss

task = args.task
mode = args.mode
if mode == 'supervised':
    prior_distribution_loss = 0
if mode == 'semi_supervised':
    prior_distribution_loss = 0
if mode == 'unsupervised':
    prior_distribution_loss = 1
    label_cross_loss = 0


if inputdata[-3:] == 'npz':
    data = sparse.load_npz(inputdata)
    data = data.todense()
else:
    data = np.load(inputdata)

if input_mask_data[-3:] == 'npz':
    input_mask_data = sparse.load_npz(input_mask_data)
    input_mask_data = input_mask_data.todense()
else:
    input_mask_data = np.load(input_mask_data)

if output_mask_data[-3:] == 'npz':
    output_mask_data = sparse.load_npz(output_mask_data)
    output_mask_data = output_mask_data.todense()
else:
    output_mask_data = np.load(output_mask_data)

batch = np.load(inputbatch)
labels = np.load(inputcelltype)

print(data.shape)
print(batch.shape)
print(labels.shape)
print(input_mask_data.shape)
print(output_mask_data.shape)

num_batch = batch.shape[1]
num_clusters = labels.shape[1]

label = np.hstack((batch,labels))
print(label.shape)

if permute_input == 'T':
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0, random_state=30)
else:
    x_train = data
    y_train = label

#print(x_train.shape)
#print(y_train.shape)

batch_information_data = y_train[:,0:num_batch]
label_information_data = y_train[:,num_batch:]

print(x_train.shape)
print(batch_information_data.shape)
print(label_information_data.shape)
print(input_mask_data.shape)
print(output_mask_data.shape)
#exit()
# model
#encoder
input_size = x_train.shape[1]
x_in = Input(shape=(input_size,))
#+ new
input_mask_shape = input_mask_data.shape[1]
input_mask = Input(shape=(input_mask_shape,))
x = Lambda(lambda x: x[0]*x[1])([x_in, input_mask])

x = Dense(intermediate_dim, activation=activation_function)(x_in)
x = Dense(z_dim, activation=activation_function)(x)

local_z_mean = Dense(z_dim)(x)
local_z_log_var = Dense(z_dim)(x)
local_encoder = Model(x_in, [local_z_mean, local_z_log_var])

z_mean, z_log_var = local_encoder(x_in)

#embedding layer
batch_information = Input(shape=(num_batch,))
if independent_embed == 'F':
    yh, _ = local_encoder(batch_information)
else:
    yh = Dense(z_dim)(batch_information)

label_information = Input(shape=(num_clusters,))

#decoder
z1 = Input(shape=(z_dim,))
h = Dense(intermediate_dim, activation=activation_function)(z1)
x_recon = Dense(input_size, activation=last_activation)(h)

decoder = Model(z1, x_recon)

#classifier
z = Input(shape=(z_dim,))
y = Dense(num_clusters, activation='softmax')(z)

classfier = Model(z, y)

#reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], z_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon

z1 = Lambda(sampling, output_shape=(z_dim,))([z_mean, z_log_var])

encoder = Model([x_in, batch_information, label_information, input_mask], [z_mean, z1, yh])

x_recon = decoder(z1)

#vector arithmetic
def arithmetics(args):
    z_local_mean, yh = args
    if arithmetic == 'minus':
        return z_local_mean - yh
    else:
        return z_local_mean + yh

z = Lambda(arithmetics, output_shape=(z_dim,))([z_mean, yh])
y = classfier(z)

#parameter for mean vector of each cluster
class Gaussian(Layer):
    def __init__(self, num_clusters, **kwargs):
        self.num_clusters = num_clusters
        super(Gaussian, self).__init__(**kwargs)

    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(name='mean',
                                    shape=(self.num_clusters, latent_dim),
                                    initializer='zeros')

    def call(self, inputs):
        z = inputs
        z = K.expand_dims(z, 1)
        return z * 0 + K.expand_dims(self.mean, 0)

    def compute_output_shape(self, input_shape):
        return (None, self.num_clusters, input_shape[-1])


gaussian = Gaussian(num_clusters)
z_prior_mean = gaussian(z)

output_mask_shape = output_mask_data.shape[1]
output_mask = Input(shape=(output_mask_shape,))
x_recon_mask = Lambda(lambda x: x[0]*x[1])([x_recon, output_mask])
x_in_mask = Lambda(lambda x: x[0]*x[1])([x_in, output_mask])
vae = Model([x_in, input_mask, batch_information, label_information,output_mask], [x_recon, z_prior_mean, y])

#vae = Model([x_in, batch_information, label_information], [x_recon, z_prior_mean, y])

#Loss
z_mean = K.expand_dims(z_mean, 1)
z_log_var = K.expand_dims(z_log_var, 1)
yh = K.expand_dims(yh, 1)

#hyper-paramter
lamb = reconstruction_loss
lamb1 = kl_cross_loss
lamb2 = prior_distribution_loss
lamb3 = label_cross_loss

label_loss = K.mean(-label_information * K.log(y + K.epsilon()), 0)
#xent_loss = 0.5 * K.mean((x_in - x_recon) ** 2, 0)
#+ new
xent_loss = 0.5 * K.mean((x_in_mask - x_recon_mask) ** 2, 0)

if arithmetic == 'minus':
    kl_loss_origin = - 0.5 * (1 + z_log_var - K.square(z_mean - z_prior_mean - yh) - K.exp(z_log_var))
else:
    kl_loss_origin = - 0.5 * (1 + z_log_var - K.square(z_mean - z_prior_mean + yh) - K.exp(z_log_var))
kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss_origin),0)
cat_loss = K.mean(y * K.log(y + K.epsilon()), 0)
vae_loss = lamb * K.sum(xent_loss) + lamb1 * K.sum(kl_loss) +lamb2*K.sum(cat_loss) + lamb3 * K.sum(label_loss)

print('vae_loss', type(vae_loss))

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

vae.summary()
Training = training
if Training == 'T':
    vae.fit([x_train, input_mask_data,batch_information_data, label_information_data,output_mask_data],
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size)

    vae.save_weights('results/training.weight')
    exit()

vae.load_weights(weights)
means = K.eval(gaussian.mean)

x_train_encoded, x_low_dimension, y = encoder.predict([data, batch, labels])
print(x_train_encoded.shape)
print(y.shape)
if arithmetic == 'minus':
    predict_x_train_encoded = x_train_encoded - y
else:
    predict_x_train_encoded = x_train_encoded + y

#np.save('results/mean_vector.npy', x_train_encoded)
#np.save('results/batch_vector.npy', y)
np.save('results/Low_dimnesion_vector.npy', x_low_dimension-y)

if task == 'imputation':
    impute_index = []
    for i in range(batch.shape[0]):
        if np.argmax(batch[i]) == 1:
            impute_index.append(i)

    x_low_dimension = x_low_dimension[np.asarray(impute_index)]
    output = decoder.predict(x_low_dimension)
    np.save('results/reconstruction.npy', y_train_pred)

if task == 'generation':
    generate_index = []
    for i in range(batch.shape[0]):
        if np.argmax(batch[i]) == 2:
            generate_index.append(i)

    x_low_dimension = x_low_dimension[np.asarray(generate_index)]
    output = decoder.predict(x_low_dimension)
    np.save('results/reconstruction.npy', y_train_pred)

    predict_x_train_encoded = predict_x_train_encoded[np.asarray(generate_index)]
    y_train_pred = classfier.predict(predict_x_train_encoded).argmax(axis=1)
    print(y_train_pred.shape)
    np.save('results/generation_predict_labels.npy', y_train_pred)