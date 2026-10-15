# Sound Unscrambler (From-Scratch PCA & FastICA)

A pure Python and NumPy implementation of Independent Component Analysis (ICA) and PCA whitening to solve the "Cocktail Party Problem"—built entirely from scratch without using any machine learning libraries like scikit-learn or PyTorch.

---

## What This Code Does
1. **Mixes Audio:** Loads clean .wav files and scrambles them together using a custom random mixing matrix.
2. **Centers Data:** Shifts the mixed data cloud's mean to zero.
3. **PCA Whitening:** Uses covariance and eigendecomposition to rotate and scale the data into a symmetric circle, stripping out linear correlation.
4. **FastICA:** Uses an iterative fixed-point loop with tanh (to maximize non-Gaussianity/spikiness) and Gram-Schmidt decorrelation to separate the mixed signals.
5. **Exports & Evaluates:** Saves the unmixed tracks as playable .wav files and prints a cross-correlation evaluation matrix against the originals.

---

## Project Structure
* `Unscrabler.py` — The main script containing the complete pipeline from scratch.
* `Sound_*.wav` — Input clean audio sources.
* `Found_Sounds_*.wav` — Output unmixed audio files.

---

## How to Run
1. Place your clean source audio files in the root directory named as `Sound_0.wav`, `Sound_1.wav`, etc.
2. Set the `sounds` variable in `Unscrabler.py` to match your number of files.
3. Run the script:
   ```bash
   python Unscrabler.py
