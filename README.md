# Cello Repertoire Classifier

A small machine learning project that classifies short cello audio clips by composer using manually selected audio features and a neural network.

This project focuses on three composers:
- **Bach**
- **Beethoven**
- **Schumann**

Rather than using broad historical periods, I narrowed the task to composer classification to create a cleaner and more feasible learning problem under limited time and data. The pipeline converts recordings into fixed-length clips, extracts spectral and tonal features with `librosa`, and trains a PyTorch multilayer perceptron (MLP) while using source-level splitting to reduce leakage between clips from the same original piece.

---

## Motivation

This project started from a reminiscence on my past. I studied cello for years and had multiple recordings of my own playing saved from earlier performances. Having also lived in Germany, where classical music and classical concert culture felt especially present in everyday life, I wanted to build something that connected that musical background with my growing interest in computing and machine learning.

I initially began with my own recordings because I wanted to use something meaningful and familiar rather than a generic dataset. That first version, however, was too limited to support a convincing classifier on its own. To make the project more feasible, I expanded the dataset with a small number of additional publicly available recordings of cello pieces by Bach, Beethoven, and Schumann. From there, I built a full pipeline that converts recordings into clips, extracts interpretable audio features, and trains a neural network to predict the composer.

More than anything, this project was an attempt to turn older musical material into something analytical and computational, not just storing recordings but actually using it towards my current interests.

---

## Project Goal

The goal of this project was to build an end-to-end music classification pipeline that:
- preprocesses raw audio recordings
- segments recordings into short clips
- extracts interpretable audio features
- avoids train/test leakage correctly
- trains and evaluates a neural network classifier

---

## Dataset

The dataset consists of cello recordings grouped into three composer classes:
- **Bach**
- **Beethoven**
- **Schumann**

Each full recording is:
1. converted to mono WAV
2. split into **10-second clips**
3. filtered to remove very quiet / near-silent clips
4. converted into a feature vector

### Why these classes?

An earlier version of the idea was to classify broad historical periods such as Baroque, Classical, and Romantic. I ultimately narrowed the task to three specific composers instead. That choice made the problem more realistic given the limited amount of data I could gather in a short time, and it also produced cleaner labels. Bach, Beethoven, and Schumann still loosely reflect different historical eras, but composer classification is a more concrete and defensible learning problem than trying to infer entire time periods from a small dataset.

### Labeling

Labels are inferred from the folder structure, for example:

```text
data/raw/bach/
data/raw/beethoven/
data/raw/schumann/
```

### source-level splitting
If clips from the same original recording appear in both train and test, results can look better than they really are.

To avoid that, I split data by source recording rather than by individual clip, so all clips from one original file stay entirely in either train or test.

## Pipeline

### 1. Convert recordings to WAV

Raw recordings are converted to:
- mono audio
- 22050 Hz sample rate

### 2. Split recordings into clips

Each recording is segmented into:
- **clip length:** 10 seconds
- **hop length:** 10 seconds

### 3. Extract features

For each clip, I extracted:
- tempo
- zero crossing rate
- RMS energy
- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral contrast
- 12 chroma features
- 6 tonnetz features
- 13 MFCC means
- 13 MFCC standard deviations

These features capture energy, timbre, brightness, and tonal structure.

### 4. Train a neural network

The feature vectors are used as input to a small PyTorch **multilayer perceptron (MLP)**.

Training includes:
- source-level train/test splitting
- feature standardization
- label encoding
- cross-entropy loss
- Adam optimizer

---

## Model

The classifier is a small feedforward neural network with:
- one input layer
- two hidden layers
- ReLU activations
- dropout
- one output layer for composer prediction

---

## Results

The final model achieved approximately:
- **test accuracy: ~79.5%**

This should be treated as a baseline result. The model also showed signs of:
- class imbalance
- confusion between some composers
- overfitting on the training set

Even so, the project demonstrates a coherent audio ML workflow and well-crafted evaluation design.

---

## Repository Structure

```text
.
├── data/
│   ├── sample_clips/     # small demo subset 
│   ├── features.csv      # extracted features
│
├── models/
│   └── model_mlp.pth
├── notebooks/
|   └── music_classification_analysis.ipynb
├── src/
│   ├── convert_audio.py
│   ├── make_clips.py
│   ├── extract_features.py
│   └── train_model.py
│
├── .gitignore
├── environment.yml
└── README.md
```
## Limitations

This project has several limitations:

- relatively small dataset
- limited number of classes
- likely class imbalance
- manually chosen features instead of end-to-end audio learning
- single train/test split

Because of these limitations, this is rather a baseline project rather than a thorough and complete version of using ML to classify composers. There are many steps to take in order to improve upon this project. Such as adding more pieces per composer, improving the class balance, comparing MLP to other models such as SVM or Random Forest, and most importantly sucessfully expanding on this to include more composer classes.

## Takeaway

This project was a compact machine learning project that I competed from scratch in audio classification. This serves as the first solo ML project I have completed, which means there is still much more room for technical growth and improvement.

More personally, it was also a way of connecting my background in cello with my interest in computation, data analysis, and machine learning.