
## Articulatory to acoustic synthesis

### Papers

* Z. C. Liu, Z. H. Ling, and L. R. Dai, “Articulatory-
to-acoustic conversion using BLSTM-RNNs with augmented
input representation,” Speech Communication, vol. 99, no.
February, pp. 161–172, 2018. [Online]. Available: https:
//doi.org/10.1016/j.specom.2018.02.008
* J. A. Gonzalez, L. A. Cheah, P. D. Green, J. M. Gilbert, S. R. Ell,
R. K. Moore, and E. Holdsworth, “Evaluation of a silent speech
interface based on magnetic sensing and deep learning for a pho-
netically rich vocabulary,” Proceedings of the Annual Conference
of the International Speech Communication Association, INTER-
SPEECH, vol. 2017-Augus, pp. 3986–3990, 2017.
* F. Taguchi and T. Kaburagi, “Articulatory-to-speech conversion
using bi-directional long short-term memory.” [Online]. Avail-
able: https://www.isca-speech.org/archive/Interspeech{\ }2018/
pdfs/0999
* CAO, Beiming, et al. Articulation-to-Speech Synthesis Using Articulatory Flesh Point Sensors' Orientation Information. In: INTERSPEECH. 2018. p. 3152-3156.  


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
  - MCD can be cheated with 0 paddding: I suspect some of the ambitious RNNs might be due to padding with zeroes

It can very well be that I missed an important detail in my code or there is a bug in my code, feel free to make a pull
request for that.


