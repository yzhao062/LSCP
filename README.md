# SCP (Selective Combination in Parallel Outlier Ensembles)
### Supplementary materials: datasets, demo source codes and sample outputs.

Zhao, Y., Hryniewicki, M.K., Nasrullah, Z., and Li, Z. SCP: Selective Combination in Parallel Outlier Ensembles. *SIAM International Conference on Data Mining (SDM)*, 2019.


**[PDF](https://http://www.cs.toronto.edu/~yuezhao/pub)** | 
**[Presentation Slides](https://www.cs.toronto.edu/~yuezhao/pub)** 

------------

Additional notes:
1. Three versions of codes are (going to be) provided:
   1. **Demo version** (demo_lof.py) is created for the fast reproduction of the experiment results. The demo version only compares the baseline algorithms with SCP algorithms. The effect of parameters, e.g., the choice of *k*, are not included.
   2. **Full version** (tba)  will be released after moderate code cleanup and optimization. In contrast to the demo version, the full version also considers the impact of parameter setting. The full version is therefore relatively slow, which will be further optimized. It is noted the demo version is sufficient to prove the idea. We suggest to using the demo version while playing with SCP, during the full version is being optimized.
   3. **Production version** (tba) will be released with full optimization and testing as a framework. The purpose of this version is to be used in real applications, which should require fewer dependencies and faster execution.
3. It is understood that there are **small variations** in the results due to the random process, e.g., spliting the training and test sets. Thus, running demo codes would only result in similar results to the paper but not the exactly same results.
------------

##  Introduction
In this paper, we propose a framework--called Selective Combination in Parallel Outlier Ensembles (SCP)--which addresses this issue by defining a local region around a test instance using the consensus of its nearest neighbors in randomly generated feature spaces.
The top-performing base detectors in this local region are selected and combined as the model's final output.

![SCP Flowchart](https://github.com/yzhao062/SCP/blob/master/figs/flowchart2.png)

## Dependency
The experiment codes are writen in Python 3.6 and built on a number of Python packages:
- numpy>=1.13
- scipy>=0.19
- scikit_learn>=0.19

Batch installation is possible using the supplied "requirements.txt" with pip or conda.

## Datasets
18 datasets are used (see dataset folder):

| Datasets   | #Sample Dimension  | Dimension  | #Outliers  | # Outlier Perc|
| -----------| ------------ | ------------ | ------------ | ------------ |
| Annthyroid | 7200  | 6  | 534   | 7.41  |
| Arrhythmia | 452   | 274 | 66   | 14.60 |
| Breastw    | 683   | 9   | 239  | 34.99 |
| Cardio     | 1831  | 21  | 176  | 9.61  |
| Glass      | 214   | 9   | 9    | 4.21  |
| Letter     | 1600  | 32  | 100  | 6.25  |
| Lympho     | 148   | 18  | 6    | 4.05  |
| Mnist      | 7603  | 100 | 700  | 9.21  |
| Musk       | 3062  | 166 | 97   | 3.17  |
| Pendigits  | 6870  | 16  | 156  | 2.27  |
| Pima       | 768   | 8   | 268  | 34.90 |
| Satellite  | 6435  | 36  | 2036 | 31.64 |
| Satimage-2 | 5803  | 36  | 71   | 1.22  |
| Shuttle    | 49097 | 9   | 3511 | 7.15  |
| Thyroid    | 3772  | 6   | 93   | 2.47  |
| Vertebral  | 240   |  6  | 30   | 12.50 |
| Vowels     | 1456  | 12  | 50   | 3.43  |
| Wbc        | 378   | 30  | 21   | 5.56  |

All datasets are accesible from http://odds.cs.stonybrook.edu/. Citation Suggestion for the datasets please refer to: 
> Shebuti Rayana (2016).  ODDS Library [http://odds.cs.stonybrook.edu]. Stony Brook, NY: Stony Brook University, Department of Computer Science.