# Deep metric learning improves lab of origin prediction of genetically engineered plasmids


This capsule provides the code associated with the article entitled "Deep Metric Learning Improves the Genetically Modified Plasmid Origin Prediction Laboratory", which implements deep metric learning using triple networks to improve state of the art in predicting those responsible for genetic modification in plasmids, and also a new methodology for interpreting these models using integrated gradients.

The reproducible run here will generate the following points:

- The triplet network model results
- The standard classification model results
- Few shot learning results (figure)
- TSNE analysis of both models (figure)

We also provide a Jupyter Notebook containing the code and figures associated with the model interpretation. However, the execution of this notebook needs a large quantity of RAM, so we decided to keep it out of the reproduction pipeline here.

All code and data necessary for a reproducible run from scratch are provided. However, performing that process will demand a long time. For obvious reasons, we don't do that inside our capsule. 

The following steps have the goal of explaining how to run the inference and training of our triplet model. 

## Environment

The environment is described through a Conda's **environment.yml**. First, you need to install  [MiniConda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://docs.anaconda.com/anaconda/install/). Then, run:

    conda env create -f environment.yml
    conda activate altlabs

This way, you'll have all the dependencies installed and will be inside the conda environment.

## Trained model

Our best model was made with a **Conv1DTripletClassifier** trained in 4 folds. The result was generated running each model with 10 TTA steps, averaging the scores of each TTA step and then averaging the scores of each of the 4 models. The trained model is inside: `output/daaefeed-3f3f-43a0-b7c2-2abf04e31e72`. The weights were saved through PyTorch-Lightning's ModelCheckpoint. The weights are the following:

    output/daaefeed-3f3f-43a0-b7c2-2abf04e31e72/tensorboard_logs_csv_logs/0_0/checkpoints/epoch=173.ckpt
    output/daaefeed-3f3f-43a0-b7c2-2abf04e31e72/tensorboard_logs_csv_logs/1_1/checkpoints/epoch=194.ckpt
    output/daaefeed-3f3f-43a0-b7c2-2abf04e31e72/tensorboard_logs_csv_logs/2_2/checkpoints/epoch=194.ckpt
    output/daaefeed-3f3f-43a0-b7c2-2abf04e31e72/tensorboard_logs_csv_logs/3_3/checkpoints/epoch=189.ckpt

To use the model, it's also necessary to load the `SentencePieceProcessor`, which is saved on `output/bpe/m1.model`. The `altlabs.dataset._BaseDataset` shows how to invoke it to encode the sequence.

For last, it's also necessary to use a dictionary called `lab_index_mapping`. This dictionary contains the mapping between each lab ID and an index used by the model (eg. 00Q4V31T has the index 0). It's also a `defaultdict` which gives the index 0 for any unknown lab. The respective lab_index_mappings of each of the 4 models are saved in a pickle file in the following paths:

    output/daaefeed-3f3f-43a0-b7c2-2abf04e31e72/tensorboard_logs_csv_logs/0_0/lab_index_mapping.pkl
    output/daaefeed-3f3f-43a0-b7c2-2abf04e31e72/tensorboard_logs_csv_logs/1_1/lab_index_mapping.pkl
    output/daaefeed-3f3f-43a0-b7c2-2abf04e31e72/tensorboard_logs_csv_logs/2_2/lab_index_mapping.pkl
    output/daaefeed-3f3f-43a0-b7c2-2abf04e31e72/tensorboard_logs_csv_logs/3_3/lab_index_mapping.pkl

The `FactorizationDataset` shows how to use the lab_index_mapping, but it's simply a dict. You can simply use `lab_index_mapping["00Q4V31T"]` to turn this ID into an index that can be used within the model.

## Training

You might notice that instead of using `train_values.csv`, we are using `train_values_grouped.csv`. It's just a preprocessed version of the dataset which we grouped sequences of each lab by their Levenshtein distance and the code to generate it is on `utils/group_sequences.py`. To make it easier, we're sending it inside the data folder.

**_NOTE:_**  To run this script you will need at least 64 GB RAM and due to the long run time you may want to split the processing by running the same script for different labs in parallel.

To train the model, just go the root directory of the project and run the following command:

    python -m altlabs.model.conv1d_triplet_classifier -kernel_sizes [1,2,3,4,5,6,7,8,9,10,11,12] -num_filters 256 -batch_size 64 -minimum_occurrences 2 -vocab_size 1001 -sequence_embedding_dim 200 -sequence_embedding True -learning_rate 1e-3 -bpe True -sequence_size_limit 1000 -reduce_on_plateau=False -split_mode=skfold -n_fold=4 -transform_sequence_fn=random_roll -dense_hidden_dim=0 -factorization_dim=200 -negative_proportion=0.0 -weight_decay=1e-5 -dropout_after_sequences=0.2 -max_epochs=200 -early_stopping_patience=20 -gpus="[0]" -tta_steps=10

A GPU is required to train this model. We trained it using a TITAN V with 11GB VRAM.

This command will not only train the model, but also generate a CSV file with the predictions. You can use that CSV file to generate the metrics. The run_triplet_experiments.ipybn shows how to use the test set to calculate the top 10 acc.