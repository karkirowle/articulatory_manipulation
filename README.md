
## Articulatory to acoustic synthesis

This paper is our developed baseline for the articulatory to acoustic synthesis and manipulation task presented in our Interspeech 2022 paper.


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


### Results

After careful experimentation, I learned the following:
- In the MCD calculation, there are many important details including:
  - The 0th power has to be excluded from the calculation
  - The MCD is calculated with respect to the modspec_smoothed signal
  - MCD can be cheated with 0 paddding: I suspect some of the ambitious MCDs might be due to padding with zeroes due to large batch training

It can very well be that I missed an important detail in my code or there is a bug in my code, feel free to make a pull
request for that.


