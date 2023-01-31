**## **This repository contains dataset and code of the "PolitePEER: Does Peer Review Hurt? A Dataset to Gauge Politeness Intensity in the Peer Reviews" Authors: Prabhat Kumar Bharti, Meith Navlakha, Mayank Agrawal, Asif Ekbal Affiliation: Indian Institute of Technology, Patna, India****

## Download the project source folder

Can download the source code using `git clone` or the `zip file`.

## Dataset:
We proffer a novel annotated dataset comprising 5 levels of politeness : (1) Highly Impolite, (2) Impolite, (3) Neutral, (4) Polite, and (5) Highly Polite. The dataset consists of `2500 review sentences`, encompassing levels of politeness tone intensity of peer reviews accrued from various multi-disciplinary venues like ICLR, NeurIPS, Publons, and ShitMyReviewersSay. However, till the paper is under internal review we have published only `500 review sentences` from our dataset for reference purposes only. 

#### NOTE: We have also stored the embeddings to expedite the training process. Since the entire dataset sums upto 68 MB, we have uploaded it [HERE](https://drive.google.com/drive/folders/1D_JuE4I17e6N0ReoCHWfRjLVMkFDBZl0?usp=sharing)

## Notebooks:

### 1) Data Pre-Processing :

This [notebook](https://github.com/meithnav/IIT-PolitenessLevels-Dataset/blob/main/notebooks/politeness-Dataset-Preprocess.ipynb) consists all the code for EDA like data cleaning, resolving data imbalance by upsampling, ohe-hot-encoding the y-labels and finally storing the embeds.

### 2) Pre-trained Embedding Model:

This [notebook](https://github.com/meithnav/IIT-PolitenessLevels-Dataset/blob/main/notebooks/politenesslevel-Pre-trainedEmbedding-model.ipynb) consists different variants of our competitve baselines analysis, wherein we feed <b> HateBERT/SciBERT/ToxicBert Embeddings</b> (either of them at a time). In the notebook `uncomment` the appropriate `name` and `embed_model_name`, the one that you want to reproduce and let the others be commented.

### 3) Custom Embedding Model:

This [notebook](https://github.com/meithnav/IIT-PolitenessLevels-Dataset/blob/main/notebooks/politeness-CustomEmbedding-model.ipynb)
consists our defined Embedding layer using word2vec. Our defined Embedding layer returns a 300 dimensional embedding vectoer which passes it to BiLSTM. NOYE: Make sure `is_BiLSTM = True` while running the notebook. Also, load the `Embedding-Matrix.pickle` [link](https://drive.google.com/file/d/1rLlHkxkujGiZNTtmoP0GplOgRVRFmCw2/view?usp=share_link) to load the weights for our custom embeddings.

### 4) Inter Annotator Agreement :
This [notebook](https://github.com/meithnav/IIT-PolitenessLevels-Dataset/blob/main/IAA/iaa.ipynb) depicts Fleiss Kappa, Krippendroff Alpha, and Cohen Kappa scores, suggesting how well multiple annotators have annoatated the review following the proposed annotation guidelines. 

#### IMPORTANT POINTS BEFORE RUNNING:

a) Change the `URL PATH` accordingly before loading the dataset(pickle files) <br>
b) The 1st tab in the notebook consists all the additional dependencies required and will be downloaded on running the cell <br>
c) For `SAVE_PATH` set the URL path where you want to save the trained model <br>

Once all the setup is complete then execute `run all`.

## Libraries & Dependencies used:

  <li>TensorFlow
  <li>Keras
  <li>Hugging Face
  <li>Matplotlib, numpy, pandas, pickle
  <li> krippendorff
