# Liquid Classification from Pouring Sounds

Classifying what liquid is being poured (beer, wine, or oil) using only the sound it makes. Audio recordings are cleaned, split into short windows, converted into spectral features, and used to train and compare four machine learning classifiers.

## Overview

Different liquids sound different when poured. Carbonation, viscosity, and flow rate all change the frequency content of the sound. This project tests whether those differences are strong enough for a classical ML model to tell the liquids apart.

The pipeline has three stages:

1. **Conversion** – raw `.m4a` recordings are converted to 16-bit PCM `.wav` at 44.1 kHz.
2. **Preprocessing and feature extraction** – audio is trimmed, filtered, segmented, and summarized into 17 numeric features per segment.
3. **Model training** – Random Forest, MLP, KNN, and SVM classifiers are trained and evaluated on the extracted features.

## Repository Structure

```
Liquid-Classification/
├── Convert_to_wav.ipynb                  # m4a -> wav conversion with ffmpeg
├── Preprocessing.ipynb                   # trimming, filtering, segmentation, feature extraction, plots
├── Model_Training.ipynb                  # RF, MLP, KNN, SVM training and evaluation
├── liquid_audio_features_filtered.csv    # extracted feature dataset (308 rows x 18 columns)
└── Dataset_wav/
    ├── Beer/                             # 20 recordings
    ├── Wine/                             # 31 recordings
    └── Oil/                              # 31 recordings
```

Folder names are used directly as class labels, so new classes can be added by creating a new folder under `Dataset_wav/`.

## Dataset

- **82 recordings** total across three classes (Beer, Wine, Oil)
- All audio resampled to **44,100 Hz**
- After segmentation: **308 samples** (Wine 121, Beer 117, Oil 70)

## Preprocessing

Implemented in `Preprocessing.ipynb`.

| Step | Details |
|------|---------|
| Trim | The first second of each recording is removed, since it was mostly background noise. |
| Band-pass filter | 5th-order Butterworth filter, 300 to 3000 Hz, applied to Beer and Wine. Oil is left unfiltered because its pouring sound overlapped heavily with background noise and filtering reduced accuracy. |
| Segmentation | 2-second windows with 50% overlap (1-second step). |
| Feature extraction | Per-segment means computed with `librosa`. |

### Features (17 total)

| Feature | Description |
|---------|-------------|
| `mfcc_1` to `mfcc_13` | Mel-frequency cepstral coefficients, capturing the overall timbre |
| `centroid` | Spectral centroid, the "center of mass" of the spectrum |
| `bandwidth` | Spectral bandwidth, the spread of frequencies around the centroid |
| `contrast` | Spectral contrast, the difference between peaks and valleys in the spectrum |
| `zcr` | Zero-crossing rate, how often the signal changes sign |

The notebook also generates figures for analysis: a before/after frequency spectrum comparing raw and filtered wine audio, and violin plots of centroid, zero-crossing rate, and contrast by liquid.

## Models and Results

Implemented in `Model_Training.ipynb`. All models use an 80/20 stratified train/test split (`random_state=42`) and standardized features.

| Model | Configuration | Test Accuracy |
|-------|---------------|---------------|
| Random Forest | 400 trees, tuned with `GridSearchCV` (5-fold) | 97% |
| MLP | Hidden layers (64, 32), ReLU, Adam, 300 iterations | 100% |
| KNN | k = 10 | 100% |
| SVM | RBF kernel, C = 1 | 100% |

### Random Forest analysis

- **Feature importance**: spectral contrast and spectral centroid were the two most informative features, followed by `mfcc_4`, `mfcc_2`, and `mfcc_3`.
- **Feature subset experiment**: retraining on the top *n* features shows F1 climbing from 0.81 (1 feature) to 0.95 (6 features) and peaking at 0.98 (16 features).
- **Overfitting checks**: 5-fold cross-validation gives a mean accuracy of 0.968. The notebook also includes a learning curve and an accuracy vs. model complexity plot across `n_estimators` and `max_depth`.

The SVM section projects the test set into 2D with PCA to visualize decision boundaries for true and predicted labels.

## Getting Started

### Requirements

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/) (only needed if converting new `.m4a` recordings)

Install the Python dependencies:

```bash
pip install librosa numpy pandas scipy scikit-learn matplotlib seaborn
```

### Running the pipeline

1. **(Optional) Convert new recordings.** Place `.m4a` files in `DatasetM4A/<Label>/` and run `Convert_to_wav.ipynb`. Output is written to `Dataset_wav/<Label>/`.
2. **Extract features.** Run `Preprocessing.ipynb`. This regenerates `liquid_audio_features_filtered.csv`.
3. **Train and evaluate models.** Run `Model_Training.ipynb`.

The feature CSV is already included, so you can skip straight to step 3.

## Limitations

- **Segment-level split.** The train/test split is done on 2-second segments, not whole recordings. Overlapping segments from the same recording can land in both sets, which likely inflates test accuracy. Splitting by recording (for example with `GroupKFold`) would give a more realistic estimate.
- **Small dataset.** 82 recordings is enough for a proof of concept but not for strong claims about generalization to new containers, rooms, or microphones.
- **Inconsistent filtering.** Because the band-pass filter is applied to Beer and Wine but not Oil, the model could partly be learning the presence or absence of filtering rather than the liquid itself.
- **Class imbalance.** Oil has fewer segments (70) than Beer and Wine.

## Future Work

- Group-aware cross-validation by recording
- Applying the same preprocessing to every class
- More recordings, more liquids, and varied recording conditions
- Data augmentation (noise, pitch shift, time stretch)
- CNNs on mel spectrograms as a comparison to hand-crafted features
