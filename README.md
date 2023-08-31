
# reef-audio-representation-learning


## Background

This project aims to develop an feature embedding extractor tailored to the domain of coral reef bioacoustic and soundscape audio recordings using self seupervised learning (SSL).

The high level workflow:
1. Train SimCLR using Pytorch on recordings of coral reefs from 19 
locations/countries
1. Extract the ResNet50 backbone from this model and asses the utility of embeddings from this using 6 held out datasets:
    1. Fish sound vs ambient background in Kenya
    1. 7x Fish sound classes in Bermuda
    1. Motorboat sound vs ambient background in Florida
    1. Bomb fishing vs ambient background in Indonesia
    1. Healthy vs degraded reefs in Australia
    1. Deep v mesophoptic reefs in French Polynesia
1. Performance of this embedding on these tasks was compared to three benchmarks:
    1. ResNet50 embedding pretrained on ImageNet
    1. VGGish audio embedding
    1. Fully trained supervised ResNet50's on each task (SSL should never be expected to exceed this)
1. These benchmarks were compared using:
    1. F1 scores from random forests trained on each task each  using the three embedding approaches were compared, alongside F1 scores from the supervised ResNet's (note not all supervised ResNet's are training stably yet)
    1. For embeddings only UMAP dimensionality reduction was performed (to 10 dims) and these reduced embeddings were clustered using affinity propagation. The fidelity of clusters to true classes was then assesed using chi-sq.
    1. 2D UMAPs were also plotted with each embedding.
    
## Folder structure
Needs further tidy up
```
|-- README.md
|-- code
|   |-- notebooks # notebooks for development
|   |   |-- embedding_extractor # more dev notebooks
|   `-- simclr-pytorch-reefs
|       |-- README.md
|       |-- colabs
|       |   |-- model_apply.ipynb
|       |   `-- read_logs.ipynb
|       |-- configs # for simclr runs
|       |   `-- reefs_configs.yaml
|       |-- environment.yml
|       |-- environment_old.yml
|       |-- evaluation # for running benchmarks
|       |   |-- batchsize_sweep.sh
|       |   |-- check_mycustomdataset.ipynb
|       |   |-- config_eval.yml
|       |   |-- embeddings
|       |   |   |-- ImageNet_embedding_extractor.ipynb
|       |   |   |-- ReefCLR_embedding_extractor.ipynb
|       |   |   |-- Results
|       |   |   |   |-- RF_results-20230831_005133.csv
|       |   |   |   `-- chisq_results-20230830_231034.csv
|       |   |   |-- VGGish_embedding_extractor.ipynb
|       |   |   |-- cluster.ipynb
|       |   |   |-- raw_embeddings
|       |   |   |   |-- #csv's of embeddings for ReefCLR, ImageNet and YAMNet
|       |   |   `-- simple_ml.ipynb
|       |   |-- fully_train_resnet_all.sh
|       |   |-- fully_train_resnet_all_augs.sh
|       |   |-- log_metrics
|       |   |   |-- # csv's of metrics from supervised resnets
|       |   |   |-- find_best_runs.ipynb
|       |   |   `-- plot_curves.ipynb
|       |   |-- model_eval.py
|       |   |-- model_states
|       |   |   `-- config.yaml
|       |   |-- model_states_i2map_simclr
|       |   |-- multiple_config_runs
|       |   |   |-- #configs for each test dataset
|       |   |-- my_custom_dataset_eval.py
|       |   |-- run_all_evals.sh
|       |   |-- train_eval.py
|       |   `-- util_eval.py
|       |-- logs
|       |-- models
|       |   |-- __init__.py
|       |   |-- encoder.py
|       |   |-- losses.py
|       |   |-- my_custom_dataset.py
|       |   |-- resnet.py
|       |   `-- ssl1.py
|       |-- myexman
|       |-- train.py
|       `-- utils
|-- data
|   `-- dataset.json
```

