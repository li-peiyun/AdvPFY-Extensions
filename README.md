# Adversarial Purification — Extended Experiments

This repo contains extended experiments based on **"Adversarial Purification with the Manifold Hypothesis" (AAAI 2024)**. The original paper's code is available at: https://github.com/GoL2022/AdvPFY

We extended the original work along four directions:

1. **Dataset**
   Implemented PGD attack, St-AE-Classifier and VAE-Classifier to additional datasets besides Fashion-MNIST in the original demo.The additional datasets are:

   - MNIST
   - SVHN
   - CIFAR-10

3. **Attack Method**
   Implemented additional attack methods and compared them with the original PGD attack. The implemented attacks are:

   - FGSM
   - BIM
   - PGD-RS
   - FGSM-L2

   We compare model performance before and after purification for each attack (original PGD vs. the new attacks).

4. **Purification Method**
    Implemented additional purification methods and compare them with the original methods. The purifications are:

    - ConvAE
    - MagNet-ConvAE
    - DAE-Recon

    We compare the results of the same adversarial samples(PGD attack).

5. **Model**

    Enhanced the ResNet50 backbone with Squeeze-and-Excitation (SE) Attention.

    - **Architecture:** Use a **ResNet-VAE** structure, jointly trained for **CIFAR-10 classification** and **image reconstruction**. The VAE structure is crucial for the manifold-based purification step.

    - **Encoder Enhancement:** The standard ResNet-50 encoder is modified by integrating **Squeeze-and-Excitation (SE) Blocks** after each of the four main residual layer sets ($\text{layer1}$ to $\text{layer4}$).

------

## Usage

### 1. Train models

1. Run [training.ipynb](./training.ipynb) to train the models.
2. Download trained model files:
   - `/content/AdvPFY/model/vae_clf.pth`
   - `/content/AdvPFY/model/stae_clf.pth`

### 2. Prepare for experiments

Upload models and files to [purify_attack.ipynb](./purify_attack.ipynb):

- Upload the generated model files `vae_clf.pth` and `stae_clf.pth` to `/content/AdvPFY/model`
- Place the custom attack implementations `my_attacks.py` in `/content/AdvPFY`
-  *(Placeholder — details to be added)*

### 3. Run attacks and purification

- Run [purify_attack.ipynb](./purify_attack.ipynb) to perform experiments.
- **Dataset Experiment:**
   - `Train-Dataset.ipynb` integrates the training process of St-AE-CLF and VAE-CLF on 4 datasets (Fashion-MNIST, MNIST, SVHN, CIFAR-10). Since different dataset has different channel settings, it has already been defined in the `Train-Dataset.ipynb` that the first 2 datasets use 'nn_model' and others use `nn_model_2`.
   - Then download the generated files ended as '_clf.pth', eg. `./fmnist_stae_clf.pth`. Upload them to `Attack_PurifyTest-Dataset.ipynb` for further purification testing.

- **Attack Method Experiment:** In order to run a specific attack, edit the notebook to comment out the default PGD attack and uncomment the call to the new attack methodyou want to evaluate.

- **Purification Method Experiment:**  You can directly run the notebook to observe the results of different purification methods.

- **Model Experiment:**  Change the original `nn_model` into `nn_model_se` using code below:

  ```
  # from model.nn_model import ResNetEnc, ResNetVAE
  from model.nn_model_se import ResNetEnc, ResNetVAE
  ```
  
  The `nn_model_se` can be find through path `model/nn_model_se.py`
  
