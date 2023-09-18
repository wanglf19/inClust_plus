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

#'''
#####################################################################################################################
parser = argparse.ArgumentParser(description='input_preparation')
# system config
parser.add_argument('--task', type=str, default='imputation_between_2_modal', help='mode for inClust+:imputation_between_2_modal, integration_paired_data, cross_modal_generation')

parser.add_argument('--inputdata1', type=str, default='data/training_data/merfish_scRNA_inputdata1.npz', help='address for input data')
parser.add_argument('--inputdata2', type=str, default='data/training_data/merfish_Merfish_inputdata2.npz', help='address for input data')
parser.add_argument('--inputdata3', type=str, default='none', help='address for input data')
parser.add_argument('--inputdata4', type=str, default='none', help='address for input data')
parser.add_argument('--inputdata_missing_modal', type=str, default='data/training_data/Fig1_merfish_data.npz', help='address for input data')
parser.add_argument('--input_covariates', type=str, default='data/training_data/merfish_input_covariates.npy', help='address for covariate (e.g. batch)')
parser.add_argument('--input_cell_types', type=str, default='data/training_data/merfish_input_cell_types.npy', help='address for celltype label')
parser.add_argument('--num_covariates', type=int, default=2, help='number of covariates')
parser.add_argument('--num_cell_types', type=int, default=13, help='number of cell types')
parser.add_argument('--imputation_index', type=str, default='data/training_data/merfish_imputation_index.npy', help='index in the data for imputation')

args = parser.parse_args()
num_covariates = args.num_covariates
num_cell_types = args.num_cell_types
input_covariates = np.load(args.input_covariates)
input_cell_types = np.load(args.input_cell_types)

task = args.task
if task == 'imputation_between_2_modal':
    inputdata1 = args.inputdata1
    if inputdata1[-3:] == 'npz':
        inputdata1 = sparse.load_npz(inputdata1)
        inputdata1 = inputdata1.todense()
    else:
        inputdata1 = np.load(inputdata1)
    inputdata2 = args.inputdata2
    if inputdata2[-3:] == 'npz':
        inputdata2 = sparse.load_npz(inputdata2)
        inputdata2 = inputdata2.todense()
    else:
        inputdata2 = np.load(inputdata2)
    imputation_index = np.load(args.imputation_index)

    if inputdata1.shape[1] != inputdata2.shape[1]:
        print('The genes in dataset1 and dataset2 must be aligned.')
        exit()

    data = np.vstack((inputdata1,inputdata2))
    print(data.shape)
    input_mask_data = np.ones_like(data)
    output_mask_data = np.ones_like(data)
    for i in range(input_mask_data.shape[0]):
        for j in range(input_mask_data.shape[1]):
            if j in imputation_index:
                input_mask_data[i, j] = 0
                if input_covariates[i] == 1:
                    output_mask_data[i, j] = 0

    sparse.save_npz('results/imputation_input_data.npz',sparse.csr_matrix(data))
    sparse.save_npz('results/imputation_input_mask.npz',sparse.csr_matrix(input_mask_data))
    sparse.save_npz('results/imputation_output_mask.npz', sparse.csr_matrix(output_mask_data))
    np.save('results/imputation_input_covariates.npy',to_categorical(input_covariates,num_covariates))
    np.save('results/imputation_input_cell_types.npy', to_categorical(input_cell_types, num_cell_types))
    exit()

if task == 'integration_paired_data':
    inputdata1 = args.inputdata1
    if inputdata1[-3:] == 'npz':
        inputdata1 = sparse.load_npz(args.inputdata1)
        inputdata1 = inputdata1.todense()
    else:
        inputdata1 = np.load(args.inputdata1)
    inputdata2 = args.inputdata2
    if inputdata2[-3:] == 'npz':
        inputdata2 = sparse.load_npz(args.inputdata2)
        inputdata2 = inputdata2.todense()
    else:
        inputdata2 = np.load(args.inputdata2)

    if inputdata1.shape[0] != inputdata2.shape[0]:
        print('The samples in paired data must be aligned.')
        exit()

    data = np.vstack((inputdata1, inputdata2))
    mask_data1_1 = np.ones_like(inputdata1)
    mask_data2_1 = np.ones_like(inputdata2)
    mask_data1_0 = np.zeros_like(inputdata1)
    mask_data2_0 = np.zeros_like(inputdata2)
    mask_data1_1_mask_data2_0 = np.vstack((mask_data1_1,mask_data2_0))
    mask_data1_0_mask_data2_1 = np.vstack((mask_data1_0,mask_data2_1))

    data = np.hstack((data,data,data,data))
    input_mask_data = np.hstack(mask_data1_0_mask_data2_1,mask_data1_1_mask_data2_0,mask_data1_0_mask_data2_1,mask_data1_1_mask_data2_0)
    output_mask_data = np.hstack(mask_data1_0_mask_data2_1,mask_data1_1_mask_data2_0,mask_data1_1_mask_data2_0,mask_data1_0_mask_data2_1)

    input_covariates = np.hstack((input_covariates,input_covariates,input_covariates,input_covariates))
    input_cell_types = np.hstack((input_cell_types,input_cell_types,input_cell_types,input_cell_types))
    sparse.save_npz('results/integration_input_data.npz', sparse.csr_matrix(data))
    sparse.save_npz('results/integration_input_mask.npz', sparse.csr_matrix(input_mask_data))
    sparse.save_npz('results/integration_output_mask.npz', sparse.csr_matrix(output_mask_data))
    np.save('results/integration_input_covariates.npy', to_categorical(input_covariates, num_covariates))
    np.save('results/integration_input_cell_types.npy', to_categorical(input_cell_types, num_cell_types))
    exit()

if task == 'integration_paired_data_with_batch_effect':
    inputdata1 = args.inputdata1
    if inputdata1[-3:] == 'npz':
        inputdata1 = sparse.load_npz(args.inputdata1)
        inputdata1 = inputdata1.todense()
    else:
        inputdata1 = np.load(args.inputdata1)
    inputdata2 = args.inputdata2
    if inputdata2[-3:] == 'npz':
        inputdata2 = sparse.load_npz(args.inputdata2)
        inputdata2 = inputdata2.todense()
    else:
        inputdata2 = np.load(args.inputdata2)

    if inputdata1.shape[0] != inputdata2.shape[0]:
        print('The samples in paired data must be aligned.')
        exit()

    inputdata3 = args.inputdata3
    if inputdata3[-3:] == 'npz':
        inputdata3 = sparse.load_npz(args.inputdata3)
        inputdata3 = inputdata3.todense()
    else:
        inputdata3 = np.load(args.inputdata3)
    inputdata4 = args.inputdata4
    if inputdata4[-3:] == 'npz':
        inputdata4 = sparse.load_npz(args.inputdata4)
        inputdata4 = inputdata4.todense()
    else:
        inputdata4 = np.load(args.inputdata4)

    if inputdata3.shape[0] != inputdata4.shape[0]:
        print('The samples in paired data must be aligned.')
        exit()

    if inputdata1.shape[1] != inputdata3.shape[1] or inputdata2.shape[1] != inputdata4.shape[1] :
        print('The elements in dataset1 and dataset2 must be aligned.')
        exit()

    data_modal1 = np.hstack((inputdata1, inputdata3))
    data_modal2 = np.hstack((inputdata2, inputdata4))
    mask_data1_1 = np.ones_like(data_modal1)
    mask_data2_1 = np.ones_like(data_modal2)
    mask_data1_0 = np.zeros_like(data_modal1)
    mask_data2_0 = np.zeros_like(data_modal2)
    mask_data1_1_mask_data2_0 = np.vstack((mask_data1_1,mask_data2_0))
    mask_data1_0_mask_data2_1 = np.vstack((mask_data1_0,mask_data2_1))
    all_data = np.vstack(data_modal1,data_modal2)
    data = np.hstack((all_data,all_data,all_data,all_data))
    input_mask_data = np.hstack(mask_data1_0_mask_data2_1,mask_data1_1_mask_data2_0,mask_data1_0_mask_data2_1,mask_data1_1_mask_data2_0)
    output_mask_data = np.hstack(mask_data1_0_mask_data2_1,mask_data1_1_mask_data2_0,mask_data1_1_mask_data2_0,mask_data1_0_mask_data2_1)

    input_covariates = np.hstack((input_covariates,input_covariates,input_covariates,input_covariates))
    input_cell_types = np.hstack((input_cell_types,input_cell_types,input_cell_types,input_cell_types))
    sparse.save_npz('results/integration_input_data.npz', sparse.csr_matrix(data))
    sparse.save_npz('results/integration_input_mask.npz', sparse.csr_matrix(input_mask_data))
    sparse.save_npz('results/integration_output_mask.npz', sparse.csr_matrix(output_mask_data))
    np.save('results/integration_input_covariates.npy', to_categorical(input_covariates, num_covariates))
    np.save('results/integration_input_cell_types.npy', to_categorical(input_cell_types, num_cell_types))
    exit()

if task == 'integration_data_with_triple_modality':
    inputdata1 = args.inputdata1
    if inputdata1[-3:] == 'npz':
        inputdata1 = sparse.load_npz(args.inputdata1)
        inputdata1 = inputdata1.todense()
    else:
        inputdata1 = np.load(args.inputdata1)
    inputdata2 = args.inputdata2
    if inputdata2[-3:] == 'npz':
        inputdata2 = sparse.load_npz(args.inputdata2)
        inputdata2 = inputdata2.todense()
    else:
        inputdata2 = np.load(args.inputdata2)

    if inputdata1.shape[0] != inputdata2.shape[0]:
        print('The samples in paired data must be aligned.')
        exit()

    inputdata3 = args.inputdata3
    if inputdata3[-3:] == 'npz':
        inputdata3 = sparse.load_npz(args.inputdata3)
        inputdata3 = inputdata3.todense()
    else:
        inputdata3 = np.load(args.inputdata3)
    inputdata4 = args.inputdata4
    if inputdata4[-3:] == 'npz':
        inputdata4 = sparse.load_npz(args.inputdata4)
        inputdata4 = inputdata4.todense()
    else:
        inputdata4 = np.load(args.inputdata4)

    if inputdata3.shape[0] != inputdata4.shape[0]:
        print('The samples in paired data must be aligned.')
        exit()

    if inputdata2.shape[1] != inputdata3.shape[1]:
        print('The elements in dataset1 and dataset2 must be aligned.')
        exit()

    psudo_inputdata5 = np.zeros((inputdata1.shape[0],inputdata3.shape[1]))
    psudo_inputdata6 = np.zeros((inputdata3.shape[0], inputdata1.shape[1]))
    mask_1_1 = np.vstack((np.ones_like(inputdata1),np.zeros_like(inputdata2),np.zeros_like(psudo_inputdata5)))
    mask_1_2 = np.vstack((np.zeros_like(inputdata1), np.ones_like(inputdata2), np.zeros_like(psudo_inputdata5)))
    mask_2_1 = np.vstack((np.zeros_like(psudo_inputdata6), np.ones_like(inputdata3), np.zeros_like(inputdata4)))
    mask_2_2 = np.vstack((np.zeros_like(psudo_inputdata6), np.zeros_like(inputdata3), np.ones_like(inputdata4)))
    data_1 = np.vstack((inputdata1,inputdata2,psudo_inputdata5))
    data_2 = np.vstack((psudo_inputdata6,inputdata3,inputdata4))

    data = np.hstack((data_1, data_1, data_2, data_2, data_1, data_2))
    input_mask_data = np.hstack((mask_1_1, mask_1_2, mask_2_1,mask_2_2,mask_1_2,mask_2_1))
    output_mask_data = np.hstack((mask_1_1, mask_1_2, mask_2_1,mask_2_2,mask_1_1,mask_2_2))
    input_covariates1 = input_covariates[0:inputdata1.shape[0]]
    input_covariates2 = input_covariates[inputdata1.shape[0]:-1]
    input_cell_types1 = input_cell_types[0:inputdata1.shape[0]]
    input_cell_types2 = input_cell_types[inputdata1.shape[0]:-1]
    input_covariates = np.hstack((input_covariates1, input_covariates1, input_covariates2, input_covariates2,input_covariates1,input_covariates2))
    input_cell_types = np.hstack((input_cell_types1, input_cell_types1, input_cell_types2, input_cell_types2,input_cell_types1,input_cell_types2))
    sparse.save_npz('results/integration_input_data.npz', sparse.csr_matrix(data))
    sparse.save_npz('results/integration_input_mask.npz', sparse.csr_matrix(input_mask_data))
    sparse.save_npz('results/integration_output_mask.npz', sparse.csr_matrix(output_mask_data))
    np.save('results/integration_input_covariates.npy', to_categorical(input_covariates, num_covariates))
    np.save('results/integration_input_cell_types.npy', to_categorical(input_cell_types, num_cell_types))
    exit()


if task == 'cross_modal_generation':
    inputdata1 = args.inputdata1
    if inputdata1[-3:] == 'npz':
        inputdata1 = sparse.load_npz(args.inputdata1)
        inputdata1 = inputdata1.todense()
    else:
        inputdata1 = np.load(args.inputdata1)
    inputdata2 = args.inputdata2
    if inputdata2[-3:] == 'npz':
        inputdata2 = sparse.load_npz(args.inputdata2)
        inputdata2 = inputdata2.todense()
    else:
        inputdata2 = np.load(args.inputdata2)
    inputdata3 = args.inputdata3
    if inputdata3[-3:] == 'npz':
        inputdata3 = sparse.load_npz(args.inputdata3)
        inputdata3 = inputdata2.todense()
    else:
        inputdata3 = np.load(args.inputdata3)

    if inputdata1.shape[0] != inputdata2.shape[0]:
        print('The samples in paired data must be aligned.')
        exit()
    if inputdata1.shape[1] != inputdata3.shape[1]:
        print('The genes must be aligned.')
        exit()

    data = np.vstack((inputdata1, inputdata2))
    data_impute = np.vstack((inputdata3, np.zeros((inputdata3.shape[0],inputdata2.shape[1]))))
    mask_data1_1 = np.ones_like(inputdata1)
    mask_data2_1 = np.ones_like(inputdata2)
    mask_data1_0 = np.zeros_like(inputdata1)
    mask_data2_0 = np.zeros_like(inputdata2)
    mask_data1_1_mask_data2_0 = np.vstack((mask_data1_1,mask_data2_0))
    mask_data1_0_mask_data2_1 = np.vstack((mask_data1_0,mask_data2_1))
    mask_impute = np.vstack((np.ones_like(inputdata3), np.zeros((inputdata3.shape[0],inputdata2.shape[1]))))

    data = np.hstack((data,data,data_impute))
    input_mask_data = np.hstack(mask_data1_1_mask_data2_0,mask_data1_1_mask_data2_0,mask_impute)
    output_mask_data = np.hstack(mask_data1_1_mask_data2_0,mask_data1_0_mask_data2_1,mask_impute)

    input_covariates = np.hstack((input_covariates,input_covariates))
    input_cell_types = np.hstack((input_cell_types,input_cell_types))
    input_covariates = to_categorical(input_covariates, num_covariates)
    input_cell_types = to_categorical(input_cell_types, num_cell_types)
    input_covariates = np.hstack((input_covariates, np.zeros((inputdata3.shape[0],num_covariates))))
    input_cell_types = np.hstack((input_cell_types, np.zeros((inputdata3.shape[0], num_cell_types))))

    sparse.save_npz('results/integration_input_data.npz', sparse.csr_matrix(data))
    sparse.save_npz('results/integration_input_mask.npz', sparse.csr_matrix(input_mask_data))
    sparse.save_npz('results/integration_output_mask.npz', sparse.csr_matrix(output_mask_data))
    np.save('results/integration_input_covariates.npy', input_covariates)
    np.save('results/integration_input_cell_types.npy', input_cell_types)
    exit()



