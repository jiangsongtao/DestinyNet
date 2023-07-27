# DestinyNet

## Description
This repository contains the code for the paper "DestinyNet: A Deep Learning-Based Single-Cell Lineage Tracing Framework for Fate Clustering, Flow, and Prediction", published in the journal Nature Biotechnology Brief Communication.

## **Introduction**

We have developed a simple deep learning framework yet is able to encode single-cell RNA sequencing data, clonal information and descendant cell types of clones to decode the fate of any undetermined cell in three ways.

<img width="574" alt="image" src="https://github.com/jiangsongtao/DestinyNet/assets/43131870/a9b1973f-4cd6-48bb-acd4-449999f84f01">

## **Model Architecture**

<img width="512" alt="image" src="https://github.com/jiangsongtao/DestinyNet/assets/43131870/3c89c2f7-bea7-4dd8-9488-8dfbe8709546">


## How to Use?

1. **Install the required environment**
    ```sh
    pip install -r requirements.txt
    ```
2. **Install the latest version of DestinyNet**
    ```sh
    pip install DestinyNet
    ```
3. **Modify the parameters in util.py, or use the default parameters**

4. **Example usage**
    ```python
    import DestinyNet
    args = DestinyNet.get_args()
    DestinyNet.train(args)
    ```

## Note
The `args` object contains all the necessary parameters for training the DestinyNet model. You can modify these parameters as per your dataset requirements.
all data used is in the https://drive.google.com/file/d/1E494DhIx5RLy0qv_6eWa9426Bfmq28po/view?usp=drive_link
