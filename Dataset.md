# Data Guideline for DestinyNet Training

## Preparing Data for DestinyNet Training

Before starting the training process for DestinyNet, it is essential to prepare your data properly. This involves constructing triplets for both training and inference based on the Ground Truth (GT) labels of your Adata data's Fate.

### Sparsity Calculation

To assess the sparsity of your data, use the `Test_Data` function as follows:

Code:
Sparsity = Test_Data(adata='Your adata path', Fate='Fate label you want to predict')

Explanation:
The `Test_Data` function calculates the proportion of cells with GT labels among all early cells in your dataset. This proportion is crucial for dynamically adjusting the Dropout rate during training. If the proportion of cells with GT labels is less than 20%, it is recommended to increase the Dropout rate to mitigate potential overfitting.

### Training Data Preparation

To prepare your training data, use the `Prepare` function:

Code:
Train = Prepare(split='train', Dropout=0, adata='Your adata path')

Explanation:
- Dropout Parameter: This is a user-defined hyperparameter, with a default value of 0. In most cases, when the number of cells with GT labels is adequate, the network performs well without additional Dropout. However, if the number of GT-labeled cells is low, it is advisable to set the Dropout rate to 0.2 to prevent overfitting. Conversely, if the number of GT-labeled cells is sufficient, setting Dropout to 0 typically yields the best performance.

### Recommendations

- Low GT Cell Count: Set Dropout to 0.2.
- Adequate GT Cell Count: Set Dropout to 0.
