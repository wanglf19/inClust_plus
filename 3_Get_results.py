#! -*- coding: utf-8 -*-
import csv
from scipy import sparse
from keras import optimizers
import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import os, csv, re
import argparse

#####################################################################################################################
parser = argparse.ArgumentParser(description='Get_results')
# system config
parser.add_argument('--task', type=str, default='imputation_between_2_modal', help='mode for inClust+:imputation_between_2_modal, integration_paired_data, cross_modal_generation')

parser.add_argument('--Low_rank_vector', type=str, default='data/training_data/merfish_scRNA_inputdata1.npz', help='address for input data')
parser.add_argument('--Reconstruction', type=str, default='data/training_data/merfish_Merfish_inputdata2.npz', help='address for input data')
parser.add_argument('--inputdata1_size', type=str, default='none', help='address for input data')
parser.add_argument('--inputdata2_size', type=str, default='none', help='address for input data')
parser.add_argument('--input_feature_size', type=str, default='data/training_data/merfish_input_covariates.npy', help='address for covariate (e.g. batch)')
parser.add_argument('--imputation_index', type=str, default='data/training_data/merfish_imputation_index.npy', help='index in the data for imputation')

args = parser.parse_args()
Low_rank_vector = np.load(args.Low_rank_vector)
task = args.task
if task == 'imputation_between_2_modal':
    reconstruction = np.load(args.Reconstruction)
    imputation_index = np.load(args.imputation_index)
    np.save('results/imputation_results.npy',reconstruction[:,imputation_index])
    np.save('results/imputation_Low_dimnesion_vector.npy', Low_dimnesion_vector)
    exit()

if task == 'integration_paired_data':
    inputdata1_size = args.inputdata1_size
    np.save('results/integration_Low_dimnesion_vector.npy',imputation_Low_dimnesion_vector[0:2*inputdata1_size,:])
    exit()

if task == 'integration_paired_data_with_batch_effect':
    inputdata1_size = args.inputdata1_size
    inputdata2_size = args.inputdata2_size
    np.save('results/integration_Low_dimnesion_vector.npy', imputation_Low_dimnesion_vector[0:2 * (inputdata1_size+inputdata2_size), :])
    exit()

if task == 'integration_data_with_triple_modality':
    inputdata1_size = args.inputdata1_size
    inputdata2_size = args.inputdata2_size
    np.save('results/integration_Low_dimnesion_vector.npy',imputation_Low_dimnesion_vector[0:2 * (inputdata1_size + inputdata2_size), :])
    exit()

if task == 'cross_modal_generation':
    reconstruction = np.load(args.Reconstruction)
    input_feature_size = args.input_feature_size
    index = reconstruction.shape[1]-input_feature_size
    np.save('results/generation_results.npy',imputation_Low_dimnesion_vector[:, index:reconstruction.shape[1]])
    exit()
