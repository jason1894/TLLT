# This the sourace code for Transfer Learning in Latent Space: Theory and Application（TLLT）

This responsity contains the implamentation of TLLS, which contains two subexperments: Control experiments and Baselines comparisons. Each subexperments has same process: Firstlt, learn the shared representation on source domain. Secondly, learn the density ratio function for the latent varables of source and target doamins. Lastly, we learn the predictor and test is performance.

## Preparations
```
# install dependent packages
pip install -r requirements.txt
```

## Data Sets
We use six data sets in this project, both include simulated data and real data. 
1. Simulated datasets 
    * [Moon](https://scikit-learn.org/0.16/modules/generated/sklearn.datasets.make_moons.html) 
    * [Circle](https://scikit-learn.org/1.5/modules/generated/sklearn.datasets.make_circles.html)
    * [Blood](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
    * [Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
    * [Haberman](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
2. Real datasets
    * [Office-Caltche](https://eecs.berkeley.edu/%20jhoffman/domainadapt/)     

**The dataset details:**

|                      | **Datasets**      | **Number-Source** | **Number-Target** | **Covariate** | **Labels** | **Features**       |
|----------------------|-------------------|-------------------|-------------------|---------------|------------|--------------------|
| **Simulated**         |                   |                   |                   |               |            |                    |
|                      | Circle            | 10000             | 10000             | 2             | 2          | (1, 32, 32)        |
|                      | Moon              | 10000             | 10000             | 2             | 2          | (1, 32, 32)        |
|                      | Breast cancer     | 365               | 318               | 9             | 2          | (1, 20, 20)        |
|                      | Blood             | 369               | 379               | 3             | 2          | (1, 20, 20)        |
|                      | Haberman          | 158               | 148               | 3             | 2          | (1, 20, 20)        |
| **Office-Caltche**    |                   |                   |                   |               |            |                    |
|                      | Caltech (C)       | 1123              | -                 | -             | 10         | (Variable)         |
|                      | DSLR (D)          | 157               | -                 | -             | 10         | (3, 1000, 1000)    |
|                 | Webcam (W)        | 295               | -                 | -             | 10         | (Variable)         |
|                      | Amazon (A)        | -                 | 958               | -             | 10         | (3, 300, 300)      |

we put some relate dataset in [Here](https://drive.google.com/drive/folders/1UZxBay6NG9wur63vc4puIV9Itw6JorcY).



For detailed discription can be found in our paper. The path "dataset/" includes the related datasets and the data process details in "data_gen.py" for simulated data sets.

## The First Subexperiment 
This subexperiment comatains the comparisons of the four methods: Method A(10), B(11), C(12) and D(13). Taking Method A on Moon data as an example, just run such commond:
```
python toy_class.py --data_name "moon" --bths 64 --epochs 200  --depth 20 --time 1 --Method "Method_A" --latdim 2 --r 0.4 --lr 3.0
```
If you want to run all methods repeatedly on Moon dataset multiple times, you can directly run script:
```
./Htoy_norm.sh
```
Other methods and data sets are similar to the above operation method.

The comparisons result has been display on the figure below

### results


| **Dataset**      | **Method-A** | **Method-B** | **Method-C** | **Method-D** |
|------------------|--------------|--------------|--------------|--------------|
| Circle           | **97.02**    | 96.40        | 95.21        | 95.70        |
| Moon             | **91.08**    | 81.55        | 84.82        | 71.72        |
| Blood            | **74.42**    | 65.98        | 62.99        | 50.62        |
| Breast cancer    | **92.19**    | 90.63        | 90.63        | 89.06        |
| Haberman         | **70.00**    | 66.67        | 66.67        | 50.00        |
| Office-Caltche   | **68.93**    | 67.19        | 66.67        | 36.98        |



## The Second Subexperiment
This subexperiment comatains the comparisons of our method with other baselins. Those baselins are
* [AFN (Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation)](https://arxiv.org/pdf/1811.07456v2)
* [BSP (Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation)](https://proceedings.mlr.press/v97/chen19i/chen19i.pdf)
* [CDAN ( Conditional Adversarial Domain Adaptation)](https://papers.nips.cc/paper_files/paper/2018/file/ab88b15733f543179858600245108dd8-Paper.pdf)
* [DANN (Unsupervised Domain Adaptation by Backpropagation)](https://proceedings.mlr.press/v37/ganin15.pdf)
* ERM
* [JAN (Deep Transfer Learning with Joint Adaptation Networks)](https://ise.thss.tsinghua.edu.cn/~mlong/doc/joint-adaptation-networks-icml17.pdf)
* [MCC (Minimum Class Confusion for Versatile Domain Adaptation)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660460.pdf)
* [MCD (Maximum Classifier Discrepancy for Unsupervised Domain Adaptation)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf)
* [MDD (Bridging Theory and Algorithm for Domain Adaptation)](https://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf)

For the above baseline methods, we directly use the excellent transfer learning library that has been integrated: [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library). To obtain the fairest comparison results possible, we set various parameters of the experiment. For details, please refer to the experimental part of our paper. It is particularly important to emphasize that we randomly initialize the network when implementing the baseline method instead of using pre-trained weights! 

We carry out this experiment on real dataset, to reproduce our result, for example, you can run
```
python office.py --source_domain "C" "D" "W" --target_domain "A" --data_name "office" --bths 64 --epochs 200 --time 1 --Method "Method_A" --latdim 128 --r 0.8 --lr 0.2
```
You also can run the script below to obatin multiple results for random initialization 
```
./Hoff_baseline.sh
```

### Comparisons Results
| **Methods**               | A → W   | D → W   | W → D   | A → D   | D → A   | W → A   |
|---------------------------|---------|---------|---------|---------|---------|---------|
| AFN                | 23.7    | 47.5    | 77.4    | 9.7     | 9.9     | 19.9    |
| BSP                  | 42.4    | 33.9    | 64.5    | 16.1    | 12.6    | 12.6    |
| CDAN                 | 13.6    | 50.8    | 35.5    | 25.8    | 9.4     | 13.6    |
| DANN                 | 30.5    | 32.2    | 32.3    | 25.8    | 16.2    | 6.8     |
| DAN                  | 22.0    | 50.8    | 51.6    | 9.7     | 5.2     | 13.1    |
| ERM                       | 11.9    | 35.6    | 38.7    | 9.7     | 7.9     | 8.4     |
| JAN                  | 28.8    | 23.7    | 45.2    | 16.1    | 15.7    | 16.8    |
| MCC                  | 10.2    | 30.5    | 51.6    | 12.9    | 9.4     | 8.9     |
| MCD                  | 11.9    | 25.4    | 48.4    | 38.7    | 18.8    | 24.1    |
| MDD                  | 11.9    | 45.8    | 61.3    | 19.4    | 9.4     | 8.9     |
| **Method A (Ours)**      | **38.98** | **42.37** | **65.63** | **34.38** | **13.54** | **24.48** |


| **Methods**                | {C,D,W} → A | {A,D,W} → C | {A,C,D} → W | {A,C,W} → D |
|----------------------------|-------------|-------------|-------------|-------------|
| AFN                   | 47.6        | 37.9        | 45.8        | 61.3        |
| BSP                  | 60.7        | 35.3        | 55.9        | 54.8        |
| CDAN                  | 61.3        | 40.2        | 57.6        | 64.5        |
| DANN                 | 59.2        | 42.0        | 64.4        | 58.1        |
| DAN                  | 59.7        | 40.2        | 40.7        | 64.5        |
| ERM                        | 59.2        | 35.3        | 52.5        | 71.0        |
| JAN                   | 56.5        | 35.7        | 54.2        | 67.7        |
| MCC                  | 64.4        | 34.4        | 62.7        | 64.5        |
| MCD                  | 48.7        | 30.8        | 40.7        | 58.1        |
| MDD                   | 52.4        | 32.6        | 44.1        | 64.5        |
| **Reweighting in latent space (Ours)**         | **76.56**   | **56.88**   | **67.80**   | **87.5**    |


Run the script below to obatin multiple results for pretrained weight parameters 
```
./Hoff_baseline_pretrained.sh
```
### Comparisons Results
|  **Methods**               | {C,D,W} → A | {A,D,W} → C | {A,C,D} → W | {A,C,W} → D |
|----------------------------|-------------|-------------|-------------|-------------|
| AFN                        | 96.3        | 92.9        | 96.6        |        96.8 |
| BSP                       | 95.3 | 86.2 | 96.8 | 96.6 |
| CDAN                      | 95.8 | 87.9 | 86.4 | 93.5 |
| DANN                      | 95.8 | 87.5 | 88.1 | 96.8 |
| DAN                       | 95.3 | 87.1 | 100.0 | 96.6 |
| ERM                       | 95.8 | 87.5 | 100.0 | 91.5 |
| JAN                       | 95.8 | 90.2 | 90.3 | 94.9 |
| MCC                       | 96.9 | 92.9 | 100.0 | 96.8 |
| MCD                       | 95.3 | 87.9 | 94.9 | 96.8 |
| MDD                       | 94.8 | 84.4 | 98.3 | 96.8 |
| **Reweighting in latent space (Ours)** | 85.9 | 76.9 | 83.4 | 100.0 |




| **Methods**               | A → W   | D → W   | W → D   | A → D   | D → A   | W → A   |
|---------------------------|---------|---------|---------|---------|---------|---------|
| AFN   | 93.2 | 100.0 | 100.0 | 100.0 | 83.8 | 86.9 |
| BSP   | 91.5 | 100.0 | 100.0 | 96.8 | 84.8 | 84.3 |
| CDAN  | 83.1 | 96.6 | 100.0 | 93.5 | 88.0 | 84.3 |
| DANN  | 81.4 | 100.0 | 100.0 | 64.5 | 88.5 | 83.8 |
| DAN   | 89.8 | 96.6 | 100.0 | 90.3 | 86.4 | 84.8 |
| ERM   | 79.7 | 81.4 | 100.0 | 83.9 | 73.3 | 78.5 |
| JAN   | 88.1 | 94.9 | 100.0 | 90.3 | 80.6 | 86.9 |
| MCC   | 96.6 | 100.0 | 100.0 | 100.0 | 86.9 | 81.7 |
| MCD   | 93.2 | 96.6 | 100.0 | 71.0 | 80.6 | 79.1 |
| MDD   | 61.0 | 91.5 | 96.8 | 67.7 | 61.8 | 58.6 |
| **Reweighting in latent space (Ours)** | 57.6 | 93.2 | 93.8 | 71.9 | 63.0 | 55.7 |
