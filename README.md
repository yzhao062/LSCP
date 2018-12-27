# LSCP (Locally Selective Combination in Parallel Outlier Ensembles)

Zhao, Y., Hryniewicki, M.K., Nasrullah, Z., and Li, Z. LSCP: Locally Selective Combination in Parallel Outlier Ensembles. 
*SIAM International Conference on Data Mining (SDM)*, 2019. **Accepted, to appear**.

Please cite the paper as:
        
    @conference{zhao2019lscp,
      title = {{LSCP}: Locally Selective Combination in Parallel Outlier Ensembles},
      author = {Zhao, Yue and Hryniewicki, Maciej K and Nasrullah, Zain and Li, Zheng},
      booktitle = {SIAM International Conference on Data Mining},
      publisher = {Society for Industrial and Applied Mathematics},
      address = {Calgary, Canada},
      month = {May},
      year = {2019}
    }

[Preprint](https://arxiv.org/abs/1812.01528) | 
[API Documentation](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.lscp) | 
[Example with PyOD](https://github.com/yzhao062/pyod/blob/master/examples/lscp_example.py) 

**Update** (Dec 25th, 2018): LSCP has been officially released in **[Python Outlier Detection (PyOD)](https://github.com/yzhao062/pyod)** V0.6.6.

**Update** (Dec 21th, 2018): LSCP has been accepted at SDM 2019. Acceptance rate 22.7% (90/397).

**Update** (Dec 6th, 2018): LSCP has been included as part of **[Python Outlier Detection (PyOD)](https://github.com/yzhao062/pyod)**, 
to be released in pyod V0.6.6.
 
------------

### Additional notes

1. Two versions of codes are provided:
   1. **Demo version** (demo_lof.py) is created for the fast reproduction of the experiment results. The demo version only compares the baseline algorithms with LSCP algorithms. The effect of parameters, e.g., the choice of *k*, are not included.
   2. **Production version** ([Python Outlier Detection (PyOD)](https://github.com/yzhao062/pyod)) is released with full optimization and testing as a framework. The purpose of this version is to be used in real applications, which should require fewer dependencies and faster execution.
2. It is understood that there are **small variations** in the results due to the random process, e.g., splitting the training and test sets. Thus, running demo codes would only result in similar results to the paper but not the exactly same results.

------------

##  Introduction
In unsupervised outlier ensembles, the absence of ground truth makes the combination of base detectors a challenging task. 
Specifically, existing parallel outlier ensembles lack a reliable way of selecting competent base detectors, 
affecting model accuracy and stability, during model combination. In this paper, 
we propose a framework--called Selective Combination in Parallel Outlier Ensembles 
(LSCP)--which addresses this issue by defining a local region around a test instance using the consensus of its nearest neighbors in randomly generated feature spaces. 
The top-performing base detectors in this local region are selected and combined as the model's final output. 
Four LSCP frameworks are proposed and compared with six widely used combination algorithms for parallel ensembles. 
Experimental results demonstrate that the Average of Maximum variant of LSCP (LSCP_AOM) consistently outperforms the baseline algorithms on most real-world datasets.

![LSCP Flowchart](https://github.com/yzhao062/LSCP/blob/master/figs/flowchart2.png)

## Dependency
The experiment codes are writen in Python 3.6 and built on a number of Python packages:
- numpy>=1.13
- numba>=0.35
- scipy>=0.19
- scikit_learn>=0.19

Batch installation is possible using the supplied "requirements.txt" with pip or conda.

````cmd
pip install -r requirements.txt
````

## Datasets
18 datasets are used (see dataset folder):

| Datasets   | #Sample Dimension  | Dimension  | #Outliers  | # Outlier Perc|
| -----------| ------------------ | ---------- | ---------- | ------------- |
| Annthyroid | 7200               | 6          | 534        | 7.41          |        
| Arrhythmia | 452                | 274        | 66         | 14.60         |
| Breastw    | 683                | 9          | 239        | 34.99         |
| Cardio     | 1831               | 21         | 176        | 9.61          |
| Glass      | 214                | 9          | 9          | 4.21          |
| Letter     | 1600               | 32         | 100        | 6.25          |
| Lympho     | 148                | 18         | 6          | 4.05          |
| Mnist      | 7603               | 100        | 700        | 9.21          |
| Musk       | 3062               | 166        | 97         | 3.17          |
| Pendigits  | 6870               | 16         | 156        | 2.27          |
| Pima       | 768                | 8          | 268        | 34.90         |
| Satellite  | 6435               | 36         | 2036       | 31.64         |
| Satimage-2 | 5803               | 36         | 71         | 1.22          |
| Shuttle    | 49097              | 9          | 3511       | 7.15          |
| Thyroid    | 3772               | 6          | 93         | 2.47          |
| Vertebral  | 240                | 6          | 30         | 12.50         |
| Vowels     | 1456               | 12         | 50         | 3.43          |
| Wbc        | 378                | 30         | 21         | 5.56          |

All datasets are accesible from http://odds.cs.stonybrook.edu/. Citation Suggestion for the datasets please refer to: 
> Shebuti Rayana (2016).  ODDS Library [http://odds.cs.stonybrook.edu]. Stony Brook, NY: Stony Brook University, Department of Computer Science.

## Usage and Sample Output (Demo Version)
Experiments could be reproduced by running **demo_lof.py** directly. You could simply download/clone the entire repository and execute the code by 

```cmd
python demo_lof.py
```
Two evaluation methods are introduced and the result would be saved into "results" folder:
1.  The area under receiver operating characteristic curve (**ROC**)
2.  mean Average Precision (**mAP**) 

## Results 

**Table 2: ROC-AUC scores (average of 20 independent trials, highest score highlighted in bold)**

| Datasets   | GG_A   | GG_M   | GG_WA  | GG_TH  | GG_AOM | GG_MOA | LSCP_A | LSCP_M | LSCP_MOA | LSCP_AOM |
| -----------| ------ | ------ | -------| ------ | -------| ------ | ------ | ------ | -------- | -------- |	
| annthyroid | 0.7679 | 0.7874 | 0.7653 | 0.7656 | 0.7827 | 0.7711 | 0.7509 | 0.7620 | **0.7924** | 0.7434 |
| arrhythmia | 0.7789 | 0.7572 | 0.7790 | 0.7317 | 0.7655 | 0.7772 | 0.7779 | 0.7743 | 0.7516 | **0.7791** |
| breastw    | 0.8662 | 0.7444 | 0.8702 | 0.8503 | 0.8338 | 0.8529 | 0.6920 | 0.8454 | 0.7158 | **0.8722** |
| cardio     | 0.9053 | 0.8876 | 0.9065 | 0.9088 | 0.9088 | 0.9125 | 0.8986 | 0.9149 | 0.8292 | **0.9250** |
| glass      | 0.7518 | 0.7582 | 0.7508 | 0.7540 | 0.7590 | 0.7556 | 0.7430 | 0.7505 | **0.7735** | 0.7510 |
| letter     | 0.7890 | **0.8546** | 0.7843 | 0.8077 | 0.8381 | 0.8031 | 0.7690 | 0.7892 | 0.8504 | 0.7685 |
| lympho     | 0.9785 | 0.9731 | 0.9776 | 0.9785 | 0.9766 | 0.9785 | 0.9782 | 0.9770 | 0.9728 | **0.9785** |
| mnist      | 0.8548 | 0.8329 | 0.8556 | 0.8250 | 0.8549 | 0.8587 | 0.8558 | 0.8612 | 0.7771 | **0.8630** |
| musk       | 0.9980 | 0.9951 | 0.9987 | 0.9987 | 0.9973 | 0.9991 | 0.9986 | 0.9963 | 0.9977 | **0.9994** |
| pendigits  | 0.8252 | 0.8414 | 0.8302 | 0.8446 | 0.8572 | 0.8417 | 0.8097 | 0.8560 | 0.7315 | **0.8615** |
| pima       | 0.6942 | 0.6468 | 0.6956 | 0.6273 | 0.6665 | 0.6904 | 0.6952 | 0.6828 | 0.6276 | **0.6972** |
| satellite  | 0.5954 | **0.6333** | 0.5950 | 0.6168 | 0.6324 | 0.6079 | 0.5912 | 0.6300 | 0.6028 | 0.6048 |
| satimage-2 | 0.9875 | 0.9906 | 0.9883 | 0.9884 | 0.9925 | 0.9913 | 0.9854 | 0.9931 | 0.9860 | **0.9938** |
| shuttle    | 0.5409 | **0.5571** | 0.5389 | 0.5506 | 0.5558 | 0.5475 | 0.5365 | 0.5544 | 0.5276 | 0.5498 |
| thyroid    | 0.9675 | 0.9346 | 0.9687 | 0.9656 | 0.9492 | 0.9652 | 0.9558 | 0.9624 | 0.9410 | **0.9693** |
| vertebral  | 0.3591 | 0.3713 | 0.3584 | 0.3839 | 0.3883 | 0.3659 | 0.3253 | 0.3798 | **0.4723** | 0.3471 |
| vowels     | 0.9117 | **0.9338** | 0.9101 | 0.9229 | 0.9284 | 0.9164 | 0.9224 | 0.9155 | 0.9261 | 0.8998 |
| wbc        | 0.9390 | 0.9313 | 0.9390 | 0.9333 | 0.9351 | 0.9391 | 0.9359 | 0.9331 | 0.9279 | **0.9400** |

**Table 3: mAP scores (average of 20 independent trials, highest score highlighted in bold)**

| Datasets   | GG_A   | GG_M   | GG_WA  | GG_TH  | GG_AOM | GG_MOA | LSCP_A | LSCP_M | LSCP_MOA | LSCP_AOM |
| -----------| ------ | ------ | -------| ------ | -------| ------ | ------ | ------ | -------- | -------- |
| annthyroid | 0.2452 | 0.2424 | 0.2460 | 0.2452 | **0.2617** | 0.2539 | 0.2379 | 0.2555 | 0.2423 | 0.2527 |
| arrhythmia | 0.3650 | 0.3516 | 0.3651 | 0.3326 | 0.3576 | 0.3650 | 0.3653 | 0.3614 | 0.3637 | **0.3680** | 
| breastw    | 0.6513 | 0.4797 | 0.6577 | 0.6335 | 0.5926 | 0.6321 | 0.4772 | 0.6110 | 0.4796 | **0.6739** |
| cardio     | 0.4260 | 0.4083 | 0.4295 | 0.4355 | 0.4496 | 0.4485 | 0.4108 | 0.4669 | 0.3399 | **0.4946** |
| glass      | 0.1397 | 0.1328 | 0.1430 | 0.1410 | 0.1340 | 0.1358 | 0.1341 | 0.1314 | **0.1479** | 0.1366 |
| letter     | 0.2323 | 0.3495 | 0.2275 | 0.2388 | 0.3018 | 0.2429 | 0.2121 | 0.2377 | **0.3682** | 0.2283 |
| lympho     | 0.8227 | 0.8001 | 0.8155 | 0.8227 | 0.8133 | 0.8227 | 0.8218 | 0.8116 | 0.7977 | **0.8300** |
| mnist      | 0.3905 | 0.3654 | 0.3913 | 0.3819 | 0.3868 | 0.3934 | 0.3914 | 0.3949 | 0.3326 | **0.3982** |
| musk       | 0.9331 | 0.8122 | 0.9536 | 0.9472 | 0.8908 | 0.9659 | 0.9365 | 0.8487 | 0.9097 | **0.9736** |
| pendigits  | 0.0690 | 0.0793 | 0.0693 | 0.0745 | 0.0820 | 0.0751 | 0.0633 | 0.0809 | 0.0573 | **0.0853** |
| pima       | 0.4901 | 0.4519 | 0.4913 | 0.4461 | 0.4662 | 0.4875 | 0.4879 | 0.4793 | 0.4366 | **0.4955** |
| satellite  | 0.4064 | **0.4447** | 0.4066 | 0.4071 | 0.4421 | 0.4167 | 0.4146 | 0.4404 | 0.4256 | 0.4208 |
| satimage-2 | 0.4092 | 0.5297 | 0.4291 | 0.4268 | 0.5998 | 0.5236 | 0.3584 | 0.6320 | 0.3801 | **0.6408** |
| shuttle    | 0.1300 | 0.1207 | 0.1295 | 0.1316 | 0.1265 | 0.1311 | 0.1203 | 0.1299 | 0.1125 | **0.1335** |
| thyroid    | 0.4257 | 0.2467 | 0.4397 | 0.4274 | 0.3338 | 0.4217 | 0.3459 | 0.3864 | 0.2449 | **0.4692** |
| vertebral  | 0.1054 | 0.1111 | 0.1053 | 0.1144 | 0.1113 | 0.1063 | 0.1003 | 0.1104 | **0.1445** | 0.1054 |
| vowels     | 0.3810 | **0.4135** | 0.3793 | 0.3835 | 0.4072 | 0.3887 | 0.4079 | 0.3938 | 0.3724 | 0.3547 |
| wbc        | 0.5536 | 0.5264 | 0.5540 | 0.5496 | 0.5412 | 0.5552 | 0.5497 | 0.5505 | 0.5315 | **0.5567** |

## Conclusions

In this work, we propose four variants of a novel unsupervised outlier detection framework called Locally Selective Combination in Parallel Outlier Ensembles (LSCP). 
Unlike traditional combination approaches, LSCP identifies the top-performing base detectors for each test instance relative to its local region. 
To validate the effectiveness of this approach, the proposed framework is assessed on 18 real-world datasets and observed to be superior to baseline algorithms. 
The ensemble approach *LSCP_AOM* demonstrated the best performance achieving the highest detection score on 11/18 datasets with respect to ROC-AUC and 12/18 datasets with respect to mAP. 
Theoretical considerations under the bias-variance framework are also provided for LSCP, alongside visualizations, to provide a holistic view of the framework. Since LSCP demonstrates the promise of data locality, we hope that future work extends this exploration by investigating the use of heterogeneous base detectors and more reliable pseudo ground truth generation methods. 
All source code, experimental results and figures used in this study are made publicly available. 
