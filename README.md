
## Articulatory to acoustic synthesis



[![DOI](https://zenodo.org/badge/469740748.svg)](https://zenodo.org/badge/latestdoi/469740748)


This repo contains our baseline for the articulatory to acoustic synthesis and manipulation task presented in our Interspeech 2022 paper. 


### Installation

To reconstruct my conda environment, please use

```
conda env create -f environment.yml
```

### Preprocessing

To run the preprocessing,

```
python preprocessing.py
```

### Train

To train the model, and save the model, run the following command:

```
python train.py
```

### Synthesis

To synthesis the test set of the MNGU corpus
```
python synthesise.py
```




