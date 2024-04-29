# Fine-grained Speech Sentiment Analysis in Chinese Psychological Support Hotlines Based on Large-scale Pre-trained Model

This repository contains material associated to this [paper](#Citation).

It contains:

- link to trained models for Negative Emotion Recognition ([link](#Trained-Negative Emotion Recognition-models))
- link to trained models for Fine-grained emotion multi-label classification ([link](#Trained-Fine-grained emotion multi-label classification-models))
- code and material for reproducing the experiments on Negative Emotion Recognition and Fine-grained emotion multi-label classification ([link](#Contents-for-reproducing-MSD-BraTS-experiments))

If you use this material, we would appreciate if you could cite the following reference.

## Citation

* Zhonglong Chen, Changwei Song, Yining Chen, Jianqiang Li, Guanghui Fu, Yongsheng Tong and Qing Zhao. Fine-grained Speech Sentiment Analysis in Chinese Psychological Support Hotlines Based on Large-scale Pre-trained Model. Preprint, todo. 

## Download and install

* To run the model, first install the Transformers library through the GitHub repo. For this example, we'll also install Datasets to load toy audio dataset from the Hugging Face Hub:

  ```
  pip install --upgrade pip
  pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]
  ```

* **Model checkpoints download**:
  Please download wav2vec 2.0 , HuBERT and Whisper models from the following links, and put these checkpoints to `models` path.

  * ```
    transformers-cli download facebook/hubert-base-ls960
    ```

  * ```
    transformers-cli download jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
    ```

  * ```
    transformers-cli download Jingmiao/whisper-small-chinese_base
    ```

  * ```
    transformers-cli download openai/whisper-small
    ```

  * ```
    transformers-cli download openai/whisper-medium
    ```

  * ```
    transformers-cli download openai/whisper-large-v3
    ```

    

## Data pre-processing

* 

## Training



## Negative Emotion Recognition models





## Fine-grained emotion multi-label classification



## Contents for reproducing  experiments

We provide the following contents for reproduction of our experiments:

- code to train models todo.
- code for inference of all models todo.
- code for computation of metrics and statistical analysis todo.



## Computation of metrics and statistical analysis

* `result`: This is the directory where the experimental results are stored, and the csv files that store the predicted values and real values.

* [`code/compute.py`](<https://github.com/czl0914/psy_hotline_analysis/blob/main/code/compute.ipynb>): This code used to calculate various evaluation indicators for multi-label experiments.Including weighted precision, weighted f1, weighted recall, and indicators of each fine-grained emotion category.

  

## References

1. A. Conneau, M. Ma, S. Khanuja, Y. Zhang, V. Axelrod, S. Dalmia, J. Riesa, C. Rivera, and A. Bapna, “Fleurs: Few-shot learning evaluation of universal representations of speech,” in 2022 IEEE Spoken Language Technology Workshop (SLT). IEEE, 2023, pp. 798–805.
2. A. Nfissi, W. Bouachir, N. Bouguila, and B. Mishara, “Unlocking the emotional states of high-risk suicide callers through speech analysis,” in 2024 IEEE 18th International Conference on Semantic Computing (ICSC). IEEE, 2024, pp. 33–40.
3. A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International Conference on Machine Learning. PMLR, 2023, pp. 28 492–28 518.
4. A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International
   conference on machine learning. PMLR, 2021, pp. 8748–8763.
5. A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “Wav2Vec 2.0: A framework for self-supervised learning of speech representations,” Advances in neural information processing systems, vol. 33, pp. 12 449–
   12 460, 2020.
