# SCP (Selective Combination in Parallel Outlier Ensembles)
### Supplementary materials: datasets, demo source codes and sample outputs.

Zhao, Y., Hryniewicki, M.K., Nasrullah, Z., and Li, Zheng. SPC: Selective Combination in Parallel Outlier Ensembles. *SIAM International Conference on Data Mining (SDM)*, 2019.


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
In this paper, we propose a framework---called Selective Combination in Parallel Outlier Ensembles (SCP)---which addresses this issue by defining a local region around a test instance using the consensus of its nearest neighbors in randomly generated feature spaces.  
The top-performing base detectors in this local region are selected and combined as the model's final output.

## Dependency
The experiment codes are writen in Python 3.6 and built on a number of Python packages:
- numpy>=1.13
- scipy>=0.19
- scikit_learn>=0.19

Batch installation is possible using the supplied "requirements.txt" with pip or conda.
