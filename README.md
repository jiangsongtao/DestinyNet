# DestinyNet

## Description
This repository contains the code for the paper "DestinyNet: A Deep Learning-Based Single-Cell Lineage Tracing Framework for Fate Clustering, Flow, and Prediction", published in the journal Nature Biotechnology Brief Communication.


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
