# ARES
### Official Implementation of ["ARES: Locally Adaptive Reconstruction-based Anomaly Scoring"](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_158.pdf)
### Adam Goodge, Bryan Hooi, Ng See Kiong and Ng Wee Siong (ECMLPKDD2022)

How can we detect anomalies: that is, samples that signifi-
cantly differ from a given set of high-dimensional data, such as images
or sensor data? This is a practical problem with numerous applications
and is also relevant to the goal of making learning algorithms more ro-
bust to unexpected inputs. Autoencoders are a popular approach, partly
due to their simplicity and their ability to perform dimension reduction.
However, the anomaly scoring function is not adaptive to the natural
variation in reconstruction error across the range of normal samples,
which hinders their ability to detect real anomalies. In this paper, we
empirically demonstrate the importance of local adaptivity for anomaly
scoring in experiments with real data. We then propose our novel Adap-
tive Reconstruction Error-based Scoring approach, which adapts its scor-
ing based on the local behaviour of reconstruction error over the latent
space. We show that this improves anomaly detection performance over
relevant baselines in a wide variety of benchmark datasets.


## Files
- main.py
- utils.py : functions for loading and pre-processing data
- model.py : contains autoencoder model, training and testing functions
- requirements.txt : packages for virtualenv

## Data
- MI-F/MI-V: https://www.kaggle.com/shasun/tool-wear-detection-in-cnc-mill
- EOPT: https://www.kaggle.com/init-owl/high-storage-system-data-for-energy-optimization
- SNSR: https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis
- OTTO: https://www.kaggle.com/c/otto-group-product-classification-challenge

## Citation
```
TBC
```
