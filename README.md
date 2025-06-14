# WGAN-GP Variants on CIFAR-10

This project explores and compares three different models of the **Wasserstein GAN with Gradient Penalty (WGAN-GP)** on the CIFAR-10 dataset

## Objective

To investigate how adding spectral normalization and dropout to the critic affects training stability and image quality.

- **Base WGAN-GP**
- **WGAN-GP + Spectral Normalization**
- **WGAN-GP + Spectral Normalization + Dropout**

## Methodology

- **Dataset**: CIFAR-10 (60,000 32x32 RGB images)
- **Framework**: PyTorch
- **Hyperparameter Tuning**: [Weights & Biases Sweeps](https://wandb.ai/)
- **Evaluation Metric**: Fréchet Inception Distance (FID)
- **Training Regime**: 10 epochs per run, 10 random sweep configs per model (30 runs total)

## Results

| Model               | Best FID Score |
|--------------------|----------------|
| **Base WGAN-GP**     | **138.46**      |
| Spectral Norm        | 149.40          |
| Spec + Dropout       | 146.76          |

The **Base model outperformed** both regularized models under the 10-epoch constraint. However, spectral normalization and dropout significantly improved **training stability**, suggesting that **longer training** may shift the outcome.
