## InClust+: the deep generative framework with mask modules for multimodal data integration, imputation, and cross-modal generation

This repository contains the official Keras implementation of:

**InClust+: the deep generative framework with mask modules for multimodal data integration, imputation, and cross-modal generation**

**Requirements**
- Python 3.6
- conda 4.4.10
- keras 2.2.4
- tensorflow 1.11.0

**Usage**
- Step 1. input_preparation: automatically generate the inputs for inClust+ (e.g. input_mask and output_mask) according to the requirements.
- Step 2. train inClust+ using the inputs generated in previous step
- Step 3. get results after training

**Step 1. input_preparation**
- Design the input_mask and output_mask according to the requirements (tasks). Users could design their own  input_mask and output_mask for their specifc task

```
#Augments:
#'--task', type=str, default='imputation_between_2_modal', help='mode for inClust+:imputation_between_2_modal, integration_paired_data, cross_modal_generation'
#'--inputdata1', type=str, default='data/training_data/merfish_scRNA_inputdata1.npz', help='address for input data'
#'--inputdata2', type=str, default='data/training_data/merfish_Merfish_inputdata2.npz', help='address for input data'
#'--inputdata3', type=str, default='data/training_data/Fig1_merfish_data.npz', help='address for input data'
#'--inputdata4', type=str, default='none', help='address for input data'
#'--inputdata_missing_modal', type=str, default='data/training_data/Fig1_merfish_data.npz', help='address for input data'
#'--input_covariates', type=str, default='data/training_data/merfish_input_covariates.npy', help='address for covariate (e.g. batch)'
#'--input_cell_types', type=str, default='data/training_data/merfish_input_cell_types.npy', help='address for celltype label'
#'--num_covariates', type=int, default=2, help='number of covariates'
#'--num_cell_types', type=int, default=13, help='number of cell types'
#'--imputation_index', type=str, default='data/training_data/merfish_imputation_index.npy', help='index in the data for imputation'
```

-automatically generation the inputs (e.g. input_mask and output_mask) according to the tasks that user selected.
- *--task = imputation_between_2_modal*

use the data from one modality (e.g. scRNA-seq) to impute the data from another modality (e.g. MERFISH).
-Inputs:
-inputdata1: the reference dataset with shape (#sample1, #feature)
-inputdata2: the dataset that would be impute, with shape (#sample2, #feature1).
-input_cell_types: the cell type of each sample, with shape (#sample1+#sample2)
-num_cell_types: the number of total cell types (int)
-imputation_index: the missing genes in inputdata2, with shape (#feature2)   *#feature = #feature1+#feature2

```
python 1_input_preparation.py --task=imputation_between_2_modal --inputdata1=data/training_data/merfish_scRNA_inputdata1.npz --inputdata2=data/training_data/merfish_Merfish_inputdata2.npz --input_cell_types=data/training_data/merfish_input_cell_types.npy --num_cell_types=13 --imputation_index=data/training_data/merfish_imputation_index.npy
```

- *--task = integration_paired_data*

integrate different modalities in the paired data
-Inputs:
-inputdata1: data of modality 1 from paired data, with shape (#sample, #feature1)
-inputdata2: data of modality 2 from paired data with shape (#sample, #feature2)
-input_cell_types: the cell type of each sample, with shape (#sample)
-num_cell_types: the number of total cell types (int)

- *--task = integration_paired_data_with_batch_effect*

integrate different modalities in two paired data with batch effects
-Inputs:
-inputdata1: data of modality 1 from paired dataset 1, with shape (#sample1, #feature1)
-inputdata2: data of modality 2 from paired dataset 1, with shape (#sample1, #feature2)
-inputdata3: data of modality 1 from paired dataset 2, with shape (#sample2, #feature1)
-inputdata4: data of modality 2 from paired dataset 2, with shape (#sample2, #feature2)
-input_cell_types: the cell type of each sample, with shape (#sample1+#sample2)
-num_cell_types: the number of total cell types (int)

- *--task = integration_data_with_triple_modality*

integrate two paired data with batch effects and one overlapped modality
-Inputs:
-inputdata1: data of modality 1 from paired dataset 1, with shape (#sample1, #feature1)
-inputdata2: data of modality 2 from paired dataset 1, with shape (#sample1, #feature2)
-inputdata3: data of modality 2 from paired dataset 2, with shape (#sample2, #feature2)
-inputdata4: data of modality 3 from paired dataset 2, with shape (#sample2, #feature3)
-input_cell_types: the cell type of each sample, with shape (#sample1+#sample2)
-num_cell_types: the number of total cell types (int)

```
python 1_input_preparation.py --task=integration_data_with_triple_modality --inputdata1=data/data_input_preparation/integration_input1.npz --inputdata2=data/data_input_preparation/integration_input2.npz --inputdata3=data/data_input_preparation/integration_input3.npz --inputdata4=data/data_input_preparation/integration_input4.npz --input_cell_types=data/data_input_preparation/integration_label.npy --num_cell_types=7
```

- *--task = cross_modal_generation*

integrate two paired data with batch effects and one overlapped modality
-Inputs:
-inputdata1: data of modality 1 from paired dataset 1, with shape (#sample1, #feature1)
-inputdata2: data of modality 2 from paired dataset 1, with shape (#sample1, #feature2)
-inputdata3: data of modality 1 from paired dataset 2, with shape (#sample2, #feature1)
-input_cell_types: the cell type of each sample, with shape (#sample1+#sample2)
-num_cell_types: the number of total cell types (int)
-input_covariates: the batch for inputdata1 and inputdata2

```
python 1_input_preparation.py --task=cross_modal_generation --inputdata1=data/data_input_preparation/generation_data1.npz --inputdata2=data/data_input_preparation/generation_data2.npz --inputdata3=data/data_input_preparation/generation_data3.npz --input_covariates=data/data_input_preparation/generation_batch.npy --input_cell_types=data/data_input_preparation/generation_label.npy --num_cell_types=30
```

**Step 2. inClust+ training**
- The training of inClust+ for all tasks is the same, just load right inputs generated from previous step.

- *About this article*
```
#Augments:
#'--inputdata', type=str, default='data/training_data/Fig1_merfish_data.npz', help='address for input data'
#'--input_covariates', type=str, default='data/training_data/Fig1_merfish_batch.npy', help='address for covariate (e.g. batch)'
#'--inputcelltype', type=str, default='data/training_data/Fig1_merfish_cell_type.npy', help='address for celltype label'
#'--input_mask', type=str, default='data/training_data/Fig1_merfish_input_mask.npy', help='address for celltype label'
#'--output_mask', type=str, default='data/training_data/Fig1_merfish_output_mask.npy', help='address for celltype label'

#'--dim_latent', type=int, default=50, help='dimension of latent space'
#'--dim_intermediate', type=int, default=200, help='dimension of intermediate layer'
#'--activation', type=str, default='relu', help='activation function: relu or tanh'
#'--last_activation', type=str, default='relu', help='activation function: relu, tanh or sigmoid'
#'--arithmetic', type=str, default='minus', help='arithmetic: minus or plus'
#'--independent_embed', type=str, default='T', help='embedding mode'

#'--batch_size', type=int, default=500, help='training parameters_batch_size'
#'--epochs', type=int, default=50, help='training parameters_epochs'

#'--training', type=str, default='T', help='training model(T) or loading model(F) '
#'--weights', type=str, default='data/weights_and_results/Fig2_demo.weight', help='trained weights'

#'--mode', type=str, default='suprevised', help='mode: supervised, semi_supervised, unsupervised, user_defined'
#'--task', type=str, default='integration', help='mode for inClust+:integration, imputation, generation'

#'--reconstruction_loss', type=int, default=5, help='The reconstruction loss for VAE'
#'--kl_cross_loss', type=int, default=1, help=''
#'--prior_distribution_loss', type=int, default=1, help='The assumption that prior distribution is uniform distribution'
#'--label_cross_loss', type=int, default=50, help='Loss for integrating label information into the model'
```

```
python 2_inClust+.py --last_activation=sigmoid --inputdata=data/training_data/imputation_input_data.npz --input_covariates=data/training_data/imputation_input_covariates.npy --inputcelltype=data/training_data/imputation_input_cell_types.npy --input_mask=data/training_data/imputation_input_mask.npz --output_mask=data/training_data/imputation_output_mask.npz
```


**Step 3. Get results**
- *For integration*
Get results from inClust+,
```
python 2_inClust+.py --last_activation=sigmoid --inputdata=data/training_data/imputation_input_data.npz --input_covariates=data/training_data/imputation_input_covariates.npy --inputcelltype=data/training_data/imputation_input_cell_types.npy --input_mask=data/training_data/imputation_input_mask.npz --output_mask=data/training_data/imputation_output_mask.npz --training=F --weights=results/training.weight
```
- *3.2 Further analysis*
```
python 3_Get_results.py --task=imputation_between_2_modal --Low_rank_vector=data/results/Low_dimnesion_vector.npy --Reconstruction=data/results/reconstruction.npy --imputation_index=data/training_data/merfish_imputation_index.npy
```

**Examples**
-example_1_cross_modal_imputation.ipynb
-example_2_cross_modal_integration.ipynb
-example_3_cross_modal_generation.ipynb

**VAE and its variant implementation**
https://github.com/bojone/vae

