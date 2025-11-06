# Adversarial Purification — Extended Experiments

This repo contains extended experiments based on **"Adversarial Purification with the Manifold Hypothesis" (AAAI 2024)**. The original paper's code is available at: https://github.com/GoL2022/AdvPFY

We extended the original work along four directions:

1. **Dataset**
    *(Placeholder — details to be added)*

2. **Attack Method**
    Implemented additional attack methods and compared them with the original PGD attack. The implemented attacks are:

   - **FGSM**
   - **BIM**
   - **PGD-RS**
   - **FGSM-L2**

   We compare model performance before and after purification for each attack (original PGD vs. the new attacks).

3. **Purification Method**
    *(Placeholder — details to be added)*

4. **Model**
    *(Placeholder — details to be added)*

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
- **Dataset Experiment:**  *(Placeholder — details to be added)*

- **Attack Method Experiment:** In order to run a specific attack, edit the notebook to comment out the default PGD attack and uncomment the call to the new attack methodyou want to evaluate.

- **Purification Method Experiment:**  *(Placeholder — details to be added)*

- **Model Experiment:**  *(Placeholder — details to be added)*

  