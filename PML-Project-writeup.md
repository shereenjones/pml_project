# PML-Project
Shereen Jones  
August 23, 2015  



## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

## The Data
The training data for this project are available here: 
     https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here: 
     https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

## Load Libraries to be used in this exercise

```r
library (ggplot2)
library (caret)
```

```
## Warning: package 'caret' was built under R version 3.2.1
```

```
## Loading required package: lattice
```

```r
library (plyr)
library (dplyr)
```

```
## Warning: package 'dplyr' was built under R version 3.2.1
```

```
## 
## Attaching package: 'dplyr'
## 
## The following objects are masked from 'package:plyr':
## 
##     arrange, count, desc, failwith, id, mutate, rename, summarise,
##     summarize
## 
## The following objects are masked from 'package:stats':
## 
##     filter, lag
## 
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library (gridExtra)
```

```
## Loading required package: grid
```

```r
library (downloader)
library (gbm)
```

```
## Warning: package 'gbm' was built under R version 3.2.2
```

```
## Loading required package: survival
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: splines
## Loading required package: parallel
## Loaded gbm 2.1.1
```

```r
library (rattle)
```

```
## Warning: package 'rattle' was built under R version 3.2.2
```

```
## Loading required package: RGtk2
```

```
## Warning: package 'RGtk2' was built under R version 3.2.2
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.5.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library (randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.2.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:dplyr':
## 
##     combine
```

## Get the Data
The files - a training file and a testing file - will be downloaded if they do not already exist in the working directory. After which the files will be loaded.


```r
fn1 <- "pml-training.csv"
URL1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
if (!file.exists(fn1)) {
     download.file (URL1, fn1, method="wininet")
}

fn2 <- "pml-testing.csv"
URL2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if (!file.exists(fn2)) {
     download.file (URL2, fn2, method="wininet")
}

traindata <- read.csv(fn1)
testdata <- read.csv(fn2)
```

## Split the training data into two data sets for training and validation purposes

```r
inTrain2 <- createDataPartition(y=traindata$classe, p=0.75, list=FALSE)
traintrain <- traindata[inTrain2,]
traintest <- traindata[-inTrain2,]
```

## Clean up the datasets

The first transformation will be to remove near zero variance variables

```r
nzv <- nearZeroVar(traindata)
## Apply transformation to all datasets
traintrain <- traintrain[, -nzv]
traintest <- traintest[, -nzv]
testdata <- testdata[, -nzv]
```

The second transformation will be to remove the high NA threshold variables


```r
highNA <- sapply(traintrain, function(x) mean(is.na(x))) > 0.95
## Apply transformation to all datasets
traintrain <- traintrain[, highNA==FALSE]
traintest <- traintest[, highNA==FALSE]
testdata <- testdata[, highNA==FALSE]
```

The third transformation will be to remove fields that have no impact on the prediction exercise - X and user.


```r
traintrain <- traintrain[, 3:length(traintrain)]
traintest <- traintest[, 3:length(traintest)]
testdata <- testdata[, 3:length(testdata)]
```

## Fit a boosted tree model

A boosted tree model is fitted to the training subset of the training data, and then validated against the test subset of the training data.  If at a suitable level of accuracy, as indicated by the confusion matrix, then this model will be applied to the test set.


```r
set.seed(72719)
model1b <- train(classe ~ ., data=traintrain, method="gbm")
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1324
##      2        1.5206             nan     0.1000    0.0885
##      3        1.4621             nan     0.1000    0.0698
##      4        1.4155             nan     0.1000    0.0554
##      5        1.3788             nan     0.1000    0.0518
##      6        1.3442             nan     0.1000    0.0470
##      7        1.3153             nan     0.1000    0.0461
##      8        1.2869             nan     0.1000    0.0397
##      9        1.2605             nan     0.1000    0.0394
##     10        1.2329             nan     0.1000    0.0309
##     20        1.0505             nan     0.1000    0.0216
##     40        0.8324             nan     0.1000    0.0136
##     60        0.6990             nan     0.1000    0.0091
##     80        0.5953             nan     0.1000    0.0083
##    100        0.5174             nan     0.1000    0.0049
##    120        0.4545             nan     0.1000    0.0033
##    140        0.4014             nan     0.1000    0.0027
##    150        0.3795             nan     0.1000    0.0035
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2012
##      2        1.4801             nan     0.1000    0.1445
##      3        1.3871             nan     0.1000    0.1078
##      4        1.3181             nan     0.1000    0.1051
##      5        1.2530             nan     0.1000    0.0977
##      6        1.1924             nan     0.1000    0.0897
##      7        1.1367             nan     0.1000    0.0724
##      8        1.0914             nan     0.1000    0.0641
##      9        1.0515             nan     0.1000    0.0531
##     10        1.0180             nan     0.1000    0.0630
##     20        0.7616             nan     0.1000    0.0363
##     40        0.4650             nan     0.1000    0.0210
##     60        0.3101             nan     0.1000    0.0102
##     80        0.2164             nan     0.1000    0.0066
##    100        0.1547             nan     0.1000    0.0039
##    120        0.1130             nan     0.1000    0.0024
##    140        0.0831             nan     0.1000    0.0013
##    150        0.0734             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2405
##      2        1.4534             nan     0.1000    0.1899
##      3        1.3325             nan     0.1000    0.1520
##      4        1.2371             nan     0.1000    0.1117
##      5        1.1651             nan     0.1000    0.1063
##      6        1.0983             nan     0.1000    0.1141
##      7        1.0299             nan     0.1000    0.0842
##      8        0.9767             nan     0.1000    0.0869
##      9        0.9245             nan     0.1000    0.0759
##     10        0.8768             nan     0.1000    0.0609
##     20        0.5738             nan     0.1000    0.0331
##     40        0.2898             nan     0.1000    0.0160
##     60        0.1635             nan     0.1000    0.0071
##     80        0.1006             nan     0.1000    0.0030
##    100        0.0675             nan     0.1000    0.0020
##    120        0.0471             nan     0.1000    0.0010
##    140        0.0338             nan     0.1000    0.0006
##    150        0.0293             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1275
##      2        1.5231             nan     0.1000    0.0900
##      3        1.4628             nan     0.1000    0.0707
##      4        1.4166             nan     0.1000    0.0549
##      5        1.3810             nan     0.1000    0.0519
##      6        1.3484             nan     0.1000    0.0475
##      7        1.3183             nan     0.1000    0.0441
##      8        1.2908             nan     0.1000    0.0405
##      9        1.2633             nan     0.1000    0.0411
##     10        1.2363             nan     0.1000    0.0342
##     20        1.0569             nan     0.1000    0.0207
##     40        0.8372             nan     0.1000    0.0127
##     60        0.7021             nan     0.1000    0.0070
##     80        0.6014             nan     0.1000    0.0043
##    100        0.5200             nan     0.1000    0.0049
##    120        0.4576             nan     0.1000    0.0051
##    140        0.4048             nan     0.1000    0.0029
##    150        0.3831             nan     0.1000    0.0040
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1901
##      2        1.4834             nan     0.1000    0.1458
##      3        1.3893             nan     0.1000    0.1158
##      4        1.3148             nan     0.1000    0.0894
##      5        1.2564             nan     0.1000    0.0842
##      6        1.2044             nan     0.1000    0.0973
##      7        1.1432             nan     0.1000    0.0793
##      8        1.0956             nan     0.1000    0.0583
##      9        1.0582             nan     0.1000    0.0574
##     10        1.0232             nan     0.1000    0.0684
##     20        0.7586             nan     0.1000    0.0251
##     40        0.4685             nan     0.1000    0.0185
##     60        0.3114             nan     0.1000    0.0075
##     80        0.2168             nan     0.1000    0.0062
##    100        0.1586             nan     0.1000    0.0046
##    120        0.1179             nan     0.1000    0.0024
##    140        0.0902             nan     0.1000    0.0015
##    150        0.0794             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2501
##      2        1.4488             nan     0.1000    0.1901
##      3        1.3265             nan     0.1000    0.1457
##      4        1.2334             nan     0.1000    0.1293
##      5        1.1517             nan     0.1000    0.1100
##      6        1.0820             nan     0.1000    0.0979
##      7        1.0210             nan     0.1000    0.0853
##      8        0.9669             nan     0.1000    0.0826
##      9        0.9169             nan     0.1000    0.0582
##     10        0.8805             nan     0.1000    0.0700
##     20        0.5810             nan     0.1000    0.0426
##     40        0.2976             nan     0.1000    0.0148
##     60        0.1672             nan     0.1000    0.0064
##     80        0.1059             nan     0.1000    0.0033
##    100        0.0712             nan     0.1000    0.0019
##    120        0.0498             nan     0.1000    0.0010
##    140        0.0360             nan     0.1000    0.0004
##    150        0.0313             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1290
##      2        1.5225             nan     0.1000    0.0889
##      3        1.4627             nan     0.1000    0.0731
##      4        1.4158             nan     0.1000    0.0554
##      5        1.3793             nan     0.1000    0.0504
##      6        1.3458             nan     0.1000    0.0476
##      7        1.3154             nan     0.1000    0.0466
##      8        1.2864             nan     0.1000    0.0403
##      9        1.2598             nan     0.1000    0.0380
##     10        1.2355             nan     0.1000    0.0326
##     20        1.0511             nan     0.1000    0.0260
##     40        0.8327             nan     0.1000    0.0118
##     60        0.6982             nan     0.1000    0.0083
##     80        0.5948             nan     0.1000    0.0041
##    100        0.5183             nan     0.1000    0.0050
##    120        0.4552             nan     0.1000    0.0042
##    140        0.4033             nan     0.1000    0.0044
##    150        0.3802             nan     0.1000    0.0028
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1950
##      2        1.4846             nan     0.1000    0.1447
##      3        1.3920             nan     0.1000    0.1123
##      4        1.3199             nan     0.1000    0.0974
##      5        1.2581             nan     0.1000    0.1002
##      6        1.1939             nan     0.1000    0.0825
##      7        1.1422             nan     0.1000    0.0749
##      8        1.0957             nan     0.1000    0.0653
##      9        1.0547             nan     0.1000    0.0609
##     10        1.0172             nan     0.1000    0.0510
##     20        0.7618             nan     0.1000    0.0323
##     40        0.4669             nan     0.1000    0.0162
##     60        0.3165             nan     0.1000    0.0064
##     80        0.2188             nan     0.1000    0.0057
##    100        0.1564             nan     0.1000    0.0042
##    120        0.1175             nan     0.1000    0.0023
##    140        0.0874             nan     0.1000    0.0014
##    150        0.0767             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2561
##      2        1.4469             nan     0.1000    0.1930
##      3        1.3266             nan     0.1000    0.1349
##      4        1.2396             nan     0.1000    0.1281
##      5        1.1593             nan     0.1000    0.1197
##      6        1.0864             nan     0.1000    0.0864
##      7        1.0305             nan     0.1000    0.0811
##      8        0.9798             nan     0.1000    0.0788
##      9        0.9320             nan     0.1000    0.0766
##     10        0.8837             nan     0.1000    0.0605
##     20        0.5835             nan     0.1000    0.0328
##     40        0.3051             nan     0.1000    0.0175
##     60        0.1701             nan     0.1000    0.0072
##     80        0.1050             nan     0.1000    0.0038
##    100        0.0680             nan     0.1000    0.0019
##    120        0.0473             nan     0.1000    0.0014
##    140        0.0333             nan     0.1000    0.0006
##    150        0.0286             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1315
##      2        1.5207             nan     0.1000    0.0924
##      3        1.4600             nan     0.1000    0.0678
##      4        1.4158             nan     0.1000    0.0571
##      5        1.3781             nan     0.1000    0.0503
##      6        1.3456             nan     0.1000    0.0462
##      7        1.3162             nan     0.1000    0.0419
##      8        1.2896             nan     0.1000    0.0425
##      9        1.2608             nan     0.1000    0.0362
##     10        1.2360             nan     0.1000    0.0315
##     20        1.0543             nan     0.1000    0.0224
##     40        0.8338             nan     0.1000    0.0116
##     60        0.7001             nan     0.1000    0.0100
##     80        0.5985             nan     0.1000    0.0055
##    100        0.5223             nan     0.1000    0.0053
##    120        0.4593             nan     0.1000    0.0044
##    140        0.4075             nan     0.1000    0.0042
##    150        0.3843             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2000
##      2        1.4802             nan     0.1000    0.1363
##      3        1.3920             nan     0.1000    0.1137
##      4        1.3178             nan     0.1000    0.1013
##      5        1.2535             nan     0.1000    0.0862
##      6        1.1985             nan     0.1000    0.0894
##      7        1.1429             nan     0.1000    0.0744
##      8        1.0960             nan     0.1000    0.0648
##      9        1.0553             nan     0.1000    0.0608
##     10        1.0178             nan     0.1000    0.0519
##     20        0.7575             nan     0.1000    0.0406
##     40        0.4751             nan     0.1000    0.0200
##     60        0.3129             nan     0.1000    0.0094
##     80        0.2234             nan     0.1000    0.0053
##    100        0.1614             nan     0.1000    0.0035
##    120        0.1200             nan     0.1000    0.0024
##    140        0.0919             nan     0.1000    0.0028
##    150        0.0796             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2498
##      2        1.4480             nan     0.1000    0.1816
##      3        1.3329             nan     0.1000    0.1474
##      4        1.2392             nan     0.1000    0.1322
##      5        1.1562             nan     0.1000    0.1078
##      6        1.0882             nan     0.1000    0.0853
##      7        1.0337             nan     0.1000    0.0914
##      8        0.9771             nan     0.1000    0.0909
##      9        0.9224             nan     0.1000    0.0749
##     10        0.8763             nan     0.1000    0.0662
##     20        0.5707             nan     0.1000    0.0298
##     40        0.2937             nan     0.1000    0.0177
##     60        0.1660             nan     0.1000    0.0063
##     80        0.1045             nan     0.1000    0.0042
##    100        0.0695             nan     0.1000    0.0017
##    120        0.0478             nan     0.1000    0.0012
##    140        0.0349             nan     0.1000    0.0007
##    150        0.0306             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1323
##      2        1.5214             nan     0.1000    0.0906
##      3        1.4617             nan     0.1000    0.0698
##      4        1.4162             nan     0.1000    0.0590
##      5        1.3784             nan     0.1000    0.0542
##      6        1.3439             nan     0.1000    0.0434
##      7        1.3156             nan     0.1000    0.0407
##      8        1.2893             nan     0.1000    0.0407
##      9        1.2610             nan     0.1000    0.0374
##     10        1.2381             nan     0.1000    0.0401
##     20        1.0482             nan     0.1000    0.0208
##     40        0.8319             nan     0.1000    0.0138
##     60        0.6972             nan     0.1000    0.0088
##     80        0.5958             nan     0.1000    0.0051
##    100        0.5171             nan     0.1000    0.0045
##    120        0.4519             nan     0.1000    0.0043
##    140        0.4012             nan     0.1000    0.0037
##    150        0.3781             nan     0.1000    0.0029
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2024
##      2        1.4799             nan     0.1000    0.1465
##      3        1.3866             nan     0.1000    0.1195
##      4        1.3112             nan     0.1000    0.0972
##      5        1.2480             nan     0.1000    0.0803
##      6        1.1970             nan     0.1000    0.0861
##      7        1.1428             nan     0.1000    0.0850
##      8        1.0908             nan     0.1000    0.0666
##      9        1.0502             nan     0.1000    0.0655
##     10        1.0100             nan     0.1000    0.0535
##     20        0.7494             nan     0.1000    0.0336
##     40        0.4615             nan     0.1000    0.0123
##     60        0.3089             nan     0.1000    0.0082
##     80        0.2169             nan     0.1000    0.0045
##    100        0.1583             nan     0.1000    0.0037
##    120        0.1177             nan     0.1000    0.0024
##    140        0.0878             nan     0.1000    0.0019
##    150        0.0769             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2652
##      2        1.4415             nan     0.1000    0.1894
##      3        1.3217             nan     0.1000    0.1479
##      4        1.2256             nan     0.1000    0.1148
##      5        1.1512             nan     0.1000    0.1073
##      6        1.0842             nan     0.1000    0.1031
##      7        1.0183             nan     0.1000    0.0836
##      8        0.9656             nan     0.1000    0.0801
##      9        0.9166             nan     0.1000    0.0788
##     10        0.8697             nan     0.1000    0.0502
##     20        0.5687             nan     0.1000    0.0449
##     40        0.2891             nan     0.1000    0.0107
##     60        0.1654             nan     0.1000    0.0064
##     80        0.0997             nan     0.1000    0.0043
##    100        0.0649             nan     0.1000    0.0021
##    120        0.0449             nan     0.1000    0.0010
##    140        0.0326             nan     0.1000    0.0004
##    150        0.0282             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1300
##      2        1.5193             nan     0.1000    0.0936
##      3        1.4581             nan     0.1000    0.0710
##      4        1.4119             nan     0.1000    0.0602
##      5        1.3733             nan     0.1000    0.0540
##      6        1.3391             nan     0.1000    0.0483
##      7        1.3085             nan     0.1000    0.0429
##      8        1.2806             nan     0.1000    0.0394
##      9        1.2543             nan     0.1000    0.0355
##     10        1.2315             nan     0.1000    0.0320
##     20        1.0490             nan     0.1000    0.0225
##     40        0.8284             nan     0.1000    0.0121
##     60        0.6910             nan     0.1000    0.0079
##     80        0.5933             nan     0.1000    0.0074
##    100        0.5155             nan     0.1000    0.0065
##    120        0.4547             nan     0.1000    0.0039
##    140        0.4034             nan     0.1000    0.0035
##    150        0.3805             nan     0.1000    0.0033
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1946
##      2        1.4813             nan     0.1000    0.1499
##      3        1.3839             nan     0.1000    0.1085
##      4        1.3137             nan     0.1000    0.1033
##      5        1.2475             nan     0.1000    0.0777
##      6        1.1969             nan     0.1000    0.0863
##      7        1.1442             nan     0.1000    0.0746
##      8        1.0975             nan     0.1000    0.0652
##      9        1.0572             nan     0.1000    0.0631
##     10        1.0182             nan     0.1000    0.0522
##     20        0.7585             nan     0.1000    0.0332
##     40        0.4675             nan     0.1000    0.0133
##     60        0.3137             nan     0.1000    0.0088
##     80        0.2215             nan     0.1000    0.0055
##    100        0.1582             nan     0.1000    0.0034
##    120        0.1169             nan     0.1000    0.0021
##    140        0.0886             nan     0.1000    0.0015
##    150        0.0773             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2574
##      2        1.4434             nan     0.1000    0.1809
##      3        1.3276             nan     0.1000    0.1491
##      4        1.2341             nan     0.1000    0.1272
##      5        1.1538             nan     0.1000    0.1043
##      6        1.0890             nan     0.1000    0.0995
##      7        1.0268             nan     0.1000    0.0794
##      8        0.9779             nan     0.1000    0.0809
##      9        0.9269             nan     0.1000    0.0690
##     10        0.8831             nan     0.1000    0.0740
##     20        0.5765             nan     0.1000    0.0408
##     40        0.2902             nan     0.1000    0.0136
##     60        0.1670             nan     0.1000    0.0066
##     80        0.1009             nan     0.1000    0.0035
##    100        0.0677             nan     0.1000    0.0014
##    120        0.0481             nan     0.1000    0.0011
##    140        0.0350             nan     0.1000    0.0007
##    150        0.0303             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1339
##      2        1.5197             nan     0.1000    0.0910
##      3        1.4600             nan     0.1000    0.0659
##      4        1.4159             nan     0.1000    0.0548
##      5        1.3795             nan     0.1000    0.0527
##      6        1.3464             nan     0.1000    0.0478
##      7        1.3152             nan     0.1000    0.0396
##      8        1.2899             nan     0.1000    0.0421
##      9        1.2619             nan     0.1000    0.0317
##     10        1.2408             nan     0.1000    0.0341
##     20        1.0570             nan     0.1000    0.0202
##     40        0.8370             nan     0.1000    0.0111
##     60        0.7009             nan     0.1000    0.0077
##     80        0.5974             nan     0.1000    0.0058
##    100        0.5171             nan     0.1000    0.0056
##    120        0.4521             nan     0.1000    0.0038
##    140        0.4041             nan     0.1000    0.0048
##    150        0.3825             nan     0.1000    0.0030
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2014
##      2        1.4801             nan     0.1000    0.1395
##      3        1.3896             nan     0.1000    0.1111
##      4        1.3176             nan     0.1000    0.0955
##      5        1.2557             nan     0.1000    0.0922
##      6        1.1972             nan     0.1000    0.0912
##      7        1.1406             nan     0.1000    0.0738
##      8        1.0962             nan     0.1000    0.0727
##      9        1.0513             nan     0.1000    0.0601
##     10        1.0138             nan     0.1000    0.0504
##     20        0.7560             nan     0.1000    0.0355
##     40        0.4682             nan     0.1000    0.0174
##     60        0.3146             nan     0.1000    0.0091
##     80        0.2203             nan     0.1000    0.0056
##    100        0.1578             nan     0.1000    0.0048
##    120        0.1140             nan     0.1000    0.0034
##    140        0.0861             nan     0.1000    0.0018
##    150        0.0758             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2581
##      2        1.4431             nan     0.1000    0.1769
##      3        1.3303             nan     0.1000    0.1386
##      4        1.2401             nan     0.1000    0.1359
##      5        1.1544             nan     0.1000    0.1109
##      6        1.0845             nan     0.1000    0.0897
##      7        1.0291             nan     0.1000    0.0952
##      8        0.9703             nan     0.1000    0.0794
##      9        0.9192             nan     0.1000    0.0767
##     10        0.8733             nan     0.1000    0.0590
##     20        0.5722             nan     0.1000    0.0399
##     40        0.2936             nan     0.1000    0.0126
##     60        0.1671             nan     0.1000    0.0067
##     80        0.1000             nan     0.1000    0.0039
##    100        0.0655             nan     0.1000    0.0018
##    120        0.0453             nan     0.1000    0.0009
##    140        0.0330             nan     0.1000    0.0008
##    150        0.0281             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1302
##      2        1.5222             nan     0.1000    0.0909
##      3        1.4629             nan     0.1000    0.0671
##      4        1.4189             nan     0.1000    0.0564
##      5        1.3824             nan     0.1000    0.0524
##      6        1.3488             nan     0.1000    0.0445
##      7        1.3201             nan     0.1000    0.0418
##      8        1.2939             nan     0.1000    0.0425
##      9        1.2659             nan     0.1000    0.0373
##     10        1.2429             nan     0.1000    0.0363
##     20        1.0614             nan     0.1000    0.0227
##     40        0.8499             nan     0.1000    0.0106
##     60        0.7110             nan     0.1000    0.0062
##     80        0.6121             nan     0.1000    0.0064
##    100        0.5320             nan     0.1000    0.0046
##    120        0.4674             nan     0.1000    0.0036
##    140        0.4172             nan     0.1000    0.0036
##    150        0.3942             nan     0.1000    0.0039
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2004
##      2        1.4815             nan     0.1000    0.1427
##      3        1.3897             nan     0.1000    0.1116
##      4        1.3176             nan     0.1000    0.0970
##      5        1.2547             nan     0.1000    0.0877
##      6        1.1989             nan     0.1000    0.0722
##      7        1.1529             nan     0.1000    0.0764
##      8        1.1052             nan     0.1000    0.0722
##      9        1.0617             nan     0.1000    0.0545
##     10        1.0275             nan     0.1000    0.0575
##     20        0.7741             nan     0.1000    0.0327
##     40        0.4771             nan     0.1000    0.0156
##     60        0.3160             nan     0.1000    0.0089
##     80        0.2223             nan     0.1000    0.0063
##    100        0.1607             nan     0.1000    0.0039
##    120        0.1188             nan     0.1000    0.0017
##    140        0.0900             nan     0.1000    0.0023
##    150        0.0790             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2543
##      2        1.4468             nan     0.1000    0.1842
##      3        1.3294             nan     0.1000    0.1476
##      4        1.2364             nan     0.1000    0.1251
##      5        1.1585             nan     0.1000    0.1061
##      6        1.0911             nan     0.1000    0.1023
##      7        1.0274             nan     0.1000    0.0846
##      8        0.9748             nan     0.1000    0.0746
##      9        0.9270             nan     0.1000    0.0748
##     10        0.8811             nan     0.1000    0.0644
##     20        0.5921             nan     0.1000    0.0398
##     40        0.3072             nan     0.1000    0.0129
##     60        0.1746             nan     0.1000    0.0083
##     80        0.1083             nan     0.1000    0.0034
##    100        0.0711             nan     0.1000    0.0029
##    120        0.0489             nan     0.1000    0.0009
##    140        0.0352             nan     0.1000    0.0008
##    150        0.0302             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1311
##      2        1.5218             nan     0.1000    0.0911
##      3        1.4615             nan     0.1000    0.0708
##      4        1.4157             nan     0.1000    0.0571
##      5        1.3785             nan     0.1000    0.0501
##      6        1.3461             nan     0.1000    0.0464
##      7        1.3166             nan     0.1000    0.0401
##      8        1.2910             nan     0.1000    0.0411
##      9        1.2631             nan     0.1000    0.0414
##     10        1.2362             nan     0.1000    0.0346
##     20        1.0511             nan     0.1000    0.0208
##     40        0.8317             nan     0.1000    0.0136
##     60        0.6967             nan     0.1000    0.0085
##     80        0.5973             nan     0.1000    0.0064
##    100        0.5190             nan     0.1000    0.0053
##    120        0.4546             nan     0.1000    0.0036
##    140        0.4042             nan     0.1000    0.0038
##    150        0.3811             nan     0.1000    0.0024
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1939
##      2        1.4807             nan     0.1000    0.1435
##      3        1.3884             nan     0.1000    0.1080
##      4        1.3189             nan     0.1000    0.1059
##      5        1.2514             nan     0.1000    0.0851
##      6        1.1970             nan     0.1000    0.0880
##      7        1.1425             nan     0.1000    0.0790
##      8        1.0937             nan     0.1000    0.0610
##      9        1.0540             nan     0.1000    0.0587
##     10        1.0172             nan     0.1000    0.0577
##     20        0.7505             nan     0.1000    0.0311
##     40        0.4645             nan     0.1000    0.0129
##     60        0.3105             nan     0.1000    0.0085
##     80        0.2138             nan     0.1000    0.0064
##    100        0.1539             nan     0.1000    0.0034
##    120        0.1130             nan     0.1000    0.0028
##    140        0.0866             nan     0.1000    0.0022
##    150        0.0761             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2633
##      2        1.4420             nan     0.1000    0.1871
##      3        1.3246             nan     0.1000    0.1510
##      4        1.2290             nan     0.1000    0.1214
##      5        1.1515             nan     0.1000    0.1026
##      6        1.0867             nan     0.1000    0.0981
##      7        1.0240             nan     0.1000    0.0887
##      8        0.9689             nan     0.1000    0.0755
##      9        0.9215             nan     0.1000    0.0833
##     10        0.8711             nan     0.1000    0.0726
##     20        0.5627             nan     0.1000    0.0374
##     40        0.2960             nan     0.1000    0.0125
##     60        0.1610             nan     0.1000    0.0057
##     80        0.1005             nan     0.1000    0.0030
##    100        0.0666             nan     0.1000    0.0023
##    120        0.0459             nan     0.1000    0.0007
##    140        0.0338             nan     0.1000    0.0008
##    150        0.0288             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1313
##      2        1.5219             nan     0.1000    0.0880
##      3        1.4642             nan     0.1000    0.0696
##      4        1.4190             nan     0.1000    0.0535
##      5        1.3825             nan     0.1000    0.0504
##      6        1.3505             nan     0.1000    0.0451
##      7        1.3216             nan     0.1000    0.0380
##      8        1.2973             nan     0.1000    0.0456
##      9        1.2675             nan     0.1000    0.0397
##     10        1.2408             nan     0.1000    0.0340
##     20        1.0592             nan     0.1000    0.0219
##     40        0.8392             nan     0.1000    0.0125
##     60        0.7038             nan     0.1000    0.0094
##     80        0.6015             nan     0.1000    0.0064
##    100        0.5209             nan     0.1000    0.0052
##    120        0.4575             nan     0.1000    0.0044
##    140        0.4058             nan     0.1000    0.0036
##    150        0.3834             nan     0.1000    0.0028
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1972
##      2        1.4839             nan     0.1000    0.1405
##      3        1.3929             nan     0.1000    0.1098
##      4        1.3211             nan     0.1000    0.0928
##      5        1.2612             nan     0.1000    0.0857
##      6        1.2064             nan     0.1000    0.0971
##      7        1.1467             nan     0.1000    0.0806
##      8        1.0969             nan     0.1000    0.0652
##      9        1.0562             nan     0.1000    0.0498
##     10        1.0241             nan     0.1000    0.0543
##     20        0.7622             nan     0.1000    0.0368
##     40        0.4660             nan     0.1000    0.0171
##     60        0.3067             nan     0.1000    0.0076
##     80        0.2109             nan     0.1000    0.0062
##    100        0.1515             nan     0.1000    0.0030
##    120        0.1111             nan     0.1000    0.0024
##    140        0.0840             nan     0.1000    0.0012
##    150        0.0743             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2512
##      2        1.4486             nan     0.1000    0.1837
##      3        1.3304             nan     0.1000    0.1511
##      4        1.2353             nan     0.1000    0.1261
##      5        1.1550             nan     0.1000    0.1007
##      6        1.0913             nan     0.1000    0.0989
##      7        1.0310             nan     0.1000    0.0822
##      8        0.9798             nan     0.1000    0.0765
##      9        0.9327             nan     0.1000    0.0660
##     10        0.8916             nan     0.1000    0.0681
##     20        0.5845             nan     0.1000    0.0381
##     40        0.2911             nan     0.1000    0.0164
##     60        0.1636             nan     0.1000    0.0091
##     80        0.0989             nan     0.1000    0.0037
##    100        0.0649             nan     0.1000    0.0019
##    120        0.0441             nan     0.1000    0.0008
##    140        0.0320             nan     0.1000    0.0005
##    150        0.0282             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1322
##      2        1.5211             nan     0.1000    0.0904
##      3        1.4614             nan     0.1000    0.0701
##      4        1.4157             nan     0.1000    0.0597
##      5        1.3768             nan     0.1000    0.0521
##      6        1.3419             nan     0.1000    0.0431
##      7        1.3134             nan     0.1000    0.0402
##      8        1.2872             nan     0.1000    0.0444
##      9        1.2582             nan     0.1000    0.0363
##     10        1.2352             nan     0.1000    0.0322
##     20        1.0500             nan     0.1000    0.0244
##     40        0.8330             nan     0.1000    0.0129
##     60        0.6966             nan     0.1000    0.0098
##     80        0.5957             nan     0.1000    0.0060
##    100        0.5173             nan     0.1000    0.0050
##    120        0.4544             nan     0.1000    0.0041
##    140        0.4010             nan     0.1000    0.0031
##    150        0.3800             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2081
##      2        1.4764             nan     0.1000    0.1392
##      3        1.3875             nan     0.1000    0.1132
##      4        1.3142             nan     0.1000    0.0970
##      5        1.2529             nan     0.1000    0.0945
##      6        1.1945             nan     0.1000    0.0874
##      7        1.1393             nan     0.1000    0.0732
##      8        1.0942             nan     0.1000    0.0763
##      9        1.0470             nan     0.1000    0.0608
##     10        1.0105             nan     0.1000    0.0512
##     20        0.7518             nan     0.1000    0.0336
##     40        0.4669             nan     0.1000    0.0175
##     60        0.3094             nan     0.1000    0.0080
##     80        0.2137             nan     0.1000    0.0054
##    100        0.1531             nan     0.1000    0.0046
##    120        0.1126             nan     0.1000    0.0020
##    140        0.0842             nan     0.1000    0.0019
##    150        0.0730             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2513
##      2        1.4485             nan     0.1000    0.1911
##      3        1.3286             nan     0.1000    0.1543
##      4        1.2321             nan     0.1000    0.1201
##      5        1.1563             nan     0.1000    0.1115
##      6        1.0869             nan     0.1000    0.0947
##      7        1.0276             nan     0.1000    0.0857
##      8        0.9753             nan     0.1000    0.0870
##      9        0.9225             nan     0.1000    0.0698
##     10        0.8801             nan     0.1000    0.0609
##     20        0.5812             nan     0.1000    0.0388
##     40        0.2907             nan     0.1000    0.0164
##     60        0.1629             nan     0.1000    0.0075
##     80        0.0988             nan     0.1000    0.0036
##    100        0.0651             nan     0.1000    0.0021
##    120        0.0445             nan     0.1000    0.0014
##    140        0.0318             nan     0.1000    0.0008
##    150        0.0269             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1283
##      2        1.5218             nan     0.1000    0.0917
##      3        1.4625             nan     0.1000    0.0688
##      4        1.4155             nan     0.1000    0.0543
##      5        1.3790             nan     0.1000    0.0507
##      6        1.3459             nan     0.1000    0.0496
##      7        1.3150             nan     0.1000    0.0453
##      8        1.2867             nan     0.1000    0.0389
##      9        1.2628             nan     0.1000    0.0390
##     10        1.2365             nan     0.1000    0.0339
##     20        1.0553             nan     0.1000    0.0248
##     40        0.8376             nan     0.1000    0.0116
##     60        0.6992             nan     0.1000    0.0096
##     80        0.6027             nan     0.1000    0.0068
##    100        0.5228             nan     0.1000    0.0061
##    120        0.4601             nan     0.1000    0.0051
##    140        0.4091             nan     0.1000    0.0031
##    150        0.3871             nan     0.1000    0.0032
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1899
##      2        1.4845             nan     0.1000    0.1394
##      3        1.3925             nan     0.1000    0.1084
##      4        1.3209             nan     0.1000    0.1021
##      5        1.2562             nan     0.1000    0.0856
##      6        1.2014             nan     0.1000    0.0929
##      7        1.1432             nan     0.1000    0.0726
##      8        1.0974             nan     0.1000    0.0727
##      9        1.0524             nan     0.1000    0.0634
##     10        1.0128             nan     0.1000    0.0556
##     20        0.7485             nan     0.1000    0.0357
##     40        0.4671             nan     0.1000    0.0172
##     60        0.3147             nan     0.1000    0.0099
##     80        0.2197             nan     0.1000    0.0060
##    100        0.1595             nan     0.1000    0.0033
##    120        0.1190             nan     0.1000    0.0024
##    140        0.0920             nan     0.1000    0.0013
##    150        0.0809             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2501
##      2        1.4470             nan     0.1000    0.1850
##      3        1.3298             nan     0.1000    0.1360
##      4        1.2411             nan     0.1000    0.1378
##      5        1.1546             nan     0.1000    0.1119
##      6        1.0863             nan     0.1000    0.0880
##      7        1.0303             nan     0.1000    0.0895
##      8        0.9755             nan     0.1000    0.0853
##      9        0.9217             nan     0.1000    0.0669
##     10        0.8793             nan     0.1000    0.0660
##     20        0.5825             nan     0.1000    0.0415
##     40        0.2947             nan     0.1000    0.0207
##     60        0.1656             nan     0.1000    0.0079
##     80        0.1032             nan     0.1000    0.0048
##    100        0.0691             nan     0.1000    0.0022
##    120        0.0490             nan     0.1000    0.0019
##    140        0.0362             nan     0.1000    0.0006
##    150        0.0316             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1269
##      2        1.5255             nan     0.1000    0.0862
##      3        1.4686             nan     0.1000    0.0676
##      4        1.4247             nan     0.1000    0.0581
##      5        1.3865             nan     0.1000    0.0503
##      6        1.3547             nan     0.1000    0.0520
##      7        1.3234             nan     0.1000    0.0481
##      8        1.2920             nan     0.1000    0.0377
##      9        1.2677             nan     0.1000    0.0397
##     10        1.2412             nan     0.1000    0.0308
##     20        1.0573             nan     0.1000    0.0206
##     40        0.8382             nan     0.1000    0.0126
##     60        0.7011             nan     0.1000    0.0079
##     80        0.6028             nan     0.1000    0.0062
##    100        0.5212             nan     0.1000    0.0045
##    120        0.4587             nan     0.1000    0.0039
##    140        0.4080             nan     0.1000    0.0038
##    150        0.3854             nan     0.1000    0.0040
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1900
##      2        1.4874             nan     0.1000    0.1516
##      3        1.3915             nan     0.1000    0.1149
##      4        1.3181             nan     0.1000    0.0858
##      5        1.2622             nan     0.1000    0.0872
##      6        1.2080             nan     0.1000    0.0865
##      7        1.1544             nan     0.1000    0.0748
##      8        1.1088             nan     0.1000    0.0627
##      9        1.0677             nan     0.1000    0.0727
##     10        1.0240             nan     0.1000    0.0555
##     20        0.7641             nan     0.1000    0.0314
##     40        0.4684             nan     0.1000    0.0188
##     60        0.3102             nan     0.1000    0.0094
##     80        0.2158             nan     0.1000    0.0055
##    100        0.1569             nan     0.1000    0.0045
##    120        0.1157             nan     0.1000    0.0039
##    140        0.0862             nan     0.1000    0.0018
##    150        0.0763             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2564
##      2        1.4432             nan     0.1000    0.1781
##      3        1.3311             nan     0.1000    0.1487
##      4        1.2365             nan     0.1000    0.1125
##      5        1.1642             nan     0.1000    0.1177
##      6        1.0891             nan     0.1000    0.0996
##      7        1.0280             nan     0.1000    0.0817
##      8        0.9774             nan     0.1000    0.0843
##      9        0.9250             nan     0.1000    0.0591
##     10        0.8872             nan     0.1000    0.0593
##     20        0.5837             nan     0.1000    0.0401
##     40        0.2998             nan     0.1000    0.0194
##     60        0.1681             nan     0.1000    0.0075
##     80        0.1031             nan     0.1000    0.0031
##    100        0.0672             nan     0.1000    0.0025
##    120        0.0462             nan     0.1000    0.0011
##    140        0.0329             nan     0.1000    0.0007
##    150        0.0279             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1306
##      2        1.5224             nan     0.1000    0.0922
##      3        1.4631             nan     0.1000    0.0714
##      4        1.4171             nan     0.1000    0.0570
##      5        1.3800             nan     0.1000    0.0519
##      6        1.3462             nan     0.1000    0.0485
##      7        1.3159             nan     0.1000    0.0477
##      8        1.2866             nan     0.1000    0.0415
##      9        1.2592             nan     0.1000    0.0368
##     10        1.2362             nan     0.1000    0.0373
##     20        1.0490             nan     0.1000    0.0191
##     40        0.8315             nan     0.1000    0.0114
##     60        0.6963             nan     0.1000    0.0096
##     80        0.5935             nan     0.1000    0.0065
##    100        0.5188             nan     0.1000    0.0048
##    120        0.4560             nan     0.1000    0.0038
##    140        0.4026             nan     0.1000    0.0027
##    150        0.3814             nan     0.1000    0.0033
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1967
##      2        1.4795             nan     0.1000    0.1416
##      3        1.3884             nan     0.1000    0.1193
##      4        1.3128             nan     0.1000    0.1055
##      5        1.2461             nan     0.1000    0.0798
##      6        1.1943             nan     0.1000    0.0879
##      7        1.1392             nan     0.1000    0.0789
##      8        1.0912             nan     0.1000    0.0646
##      9        1.0501             nan     0.1000    0.0638
##     10        1.0119             nan     0.1000    0.0549
##     20        0.7509             nan     0.1000    0.0296
##     40        0.4586             nan     0.1000    0.0196
##     60        0.3094             nan     0.1000    0.0095
##     80        0.2153             nan     0.1000    0.0051
##    100        0.1551             nan     0.1000    0.0035
##    120        0.1136             nan     0.1000    0.0026
##    140        0.0861             nan     0.1000    0.0017
##    150        0.0747             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2564
##      2        1.4434             nan     0.1000    0.1879
##      3        1.3250             nan     0.1000    0.1527
##      4        1.2301             nan     0.1000    0.1222
##      5        1.1537             nan     0.1000    0.1093
##      6        1.0848             nan     0.1000    0.0989
##      7        1.0214             nan     0.1000    0.0900
##      8        0.9669             nan     0.1000    0.0764
##      9        0.9184             nan     0.1000    0.0724
##     10        0.8742             nan     0.1000    0.0646
##     20        0.5684             nan     0.1000    0.0386
##     40        0.2881             nan     0.1000    0.0153
##     60        0.1634             nan     0.1000    0.0075
##     80        0.0993             nan     0.1000    0.0026
##    100        0.0652             nan     0.1000    0.0017
##    120        0.0473             nan     0.1000    0.0016
##    140        0.0349             nan     0.1000    0.0007
##    150        0.0292             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1380
##      2        1.5178             nan     0.1000    0.0923
##      3        1.4570             nan     0.1000    0.0702
##      4        1.4116             nan     0.1000    0.0591
##      5        1.3739             nan     0.1000    0.0475
##      6        1.3425             nan     0.1000    0.0425
##      7        1.3151             nan     0.1000    0.0475
##      8        1.2860             nan     0.1000    0.0437
##      9        1.2578             nan     0.1000    0.0389
##     10        1.2330             nan     0.1000    0.0385
##     20        1.0481             nan     0.1000    0.0226
##     40        0.8290             nan     0.1000    0.0130
##     60        0.6924             nan     0.1000    0.0089
##     80        0.5950             nan     0.1000    0.0066
##    100        0.5165             nan     0.1000    0.0061
##    120        0.4527             nan     0.1000    0.0037
##    140        0.4018             nan     0.1000    0.0024
##    150        0.3795             nan     0.1000    0.0030
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2036
##      2        1.4805             nan     0.1000    0.1404
##      3        1.3894             nan     0.1000    0.1137
##      4        1.3144             nan     0.1000    0.1027
##      5        1.2493             nan     0.1000    0.0858
##      6        1.1956             nan     0.1000    0.0842
##      7        1.1432             nan     0.1000    0.0759
##      8        1.0962             nan     0.1000    0.0734
##      9        1.0516             nan     0.1000    0.0613
##     10        1.0144             nan     0.1000    0.0563
##     20        0.7561             nan     0.1000    0.0350
##     40        0.4649             nan     0.1000    0.0156
##     60        0.3094             nan     0.1000    0.0096
##     80        0.2152             nan     0.1000    0.0055
##    100        0.1554             nan     0.1000    0.0029
##    120        0.1149             nan     0.1000    0.0032
##    140        0.0864             nan     0.1000    0.0017
##    150        0.0742             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2557
##      2        1.4459             nan     0.1000    0.1885
##      3        1.3257             nan     0.1000    0.1332
##      4        1.2393             nan     0.1000    0.1350
##      5        1.1528             nan     0.1000    0.1149
##      6        1.0824             nan     0.1000    0.0849
##      7        1.0267             nan     0.1000    0.0791
##      8        0.9774             nan     0.1000    0.0798
##      9        0.9280             nan     0.1000    0.0821
##     10        0.8775             nan     0.1000    0.0694
##     20        0.5750             nan     0.1000    0.0317
##     40        0.2941             nan     0.1000    0.0162
##     60        0.1646             nan     0.1000    0.0080
##     80        0.0976             nan     0.1000    0.0038
##    100        0.0658             nan     0.1000    0.0019
##    120        0.0449             nan     0.1000    0.0010
##    140        0.0323             nan     0.1000    0.0006
##    150        0.0280             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1307
##      2        1.5221             nan     0.1000    0.0916
##      3        1.4620             nan     0.1000    0.0688
##      4        1.4157             nan     0.1000    0.0567
##      5        1.3782             nan     0.1000    0.0514
##      6        1.3445             nan     0.1000    0.0520
##      7        1.3124             nan     0.1000    0.0390
##      8        1.2875             nan     0.1000    0.0431
##      9        1.2585             nan     0.1000    0.0361
##     10        1.2360             nan     0.1000    0.0358
##     20        1.0511             nan     0.1000    0.0230
##     40        0.8308             nan     0.1000    0.0120
##     60        0.6954             nan     0.1000    0.0084
##     80        0.5977             nan     0.1000    0.0068
##    100        0.5200             nan     0.1000    0.0047
##    120        0.4547             nan     0.1000    0.0030
##    140        0.4055             nan     0.1000    0.0028
##    150        0.3825             nan     0.1000    0.0031
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1955
##      2        1.4815             nan     0.1000    0.1431
##      3        1.3888             nan     0.1000    0.1142
##      4        1.3166             nan     0.1000    0.0958
##      5        1.2543             nan     0.1000    0.1001
##      6        1.1925             nan     0.1000    0.0691
##      7        1.1480             nan     0.1000    0.0831
##      8        1.0958             nan     0.1000    0.0671
##      9        1.0546             nan     0.1000    0.0620
##     10        1.0151             nan     0.1000    0.0476
##     20        0.7602             nan     0.1000    0.0311
##     40        0.4689             nan     0.1000    0.0165
##     60        0.3156             nan     0.1000    0.0085
##     80        0.2227             nan     0.1000    0.0057
##    100        0.1614             nan     0.1000    0.0042
##    120        0.1180             nan     0.1000    0.0039
##    140        0.0896             nan     0.1000    0.0016
##    150        0.0781             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2642
##      2        1.4437             nan     0.1000    0.1901
##      3        1.3224             nan     0.1000    0.1496
##      4        1.2282             nan     0.1000    0.1194
##      5        1.1520             nan     0.1000    0.1189
##      6        1.0776             nan     0.1000    0.0947
##      7        1.0179             nan     0.1000    0.0889
##      8        0.9633             nan     0.1000    0.0678
##      9        0.9198             nan     0.1000    0.0768
##     10        0.8738             nan     0.1000    0.0602
##     20        0.5745             nan     0.1000    0.0326
##     40        0.2911             nan     0.1000    0.0184
##     60        0.1653             nan     0.1000    0.0061
##     80        0.1029             nan     0.1000    0.0026
##    100        0.0677             nan     0.1000    0.0022
##    120        0.0474             nan     0.1000    0.0012
##    140        0.0341             nan     0.1000    0.0012
##    150        0.0292             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1305
##      2        1.5213             nan     0.1000    0.0863
##      3        1.4631             nan     0.1000    0.0676
##      4        1.4185             nan     0.1000    0.0579
##      5        1.3814             nan     0.1000    0.0495
##      6        1.3494             nan     0.1000    0.0489
##      7        1.3182             nan     0.1000    0.0398
##      8        1.2926             nan     0.1000    0.0419
##      9        1.2651             nan     0.1000    0.0378
##     10        1.2415             nan     0.1000    0.0383
##     20        1.0597             nan     0.1000    0.0207
##     40        0.8421             nan     0.1000    0.0101
##     60        0.7065             nan     0.1000    0.0069
##     80        0.6056             nan     0.1000    0.0071
##    100        0.5282             nan     0.1000    0.0046
##    120        0.4660             nan     0.1000    0.0029
##    140        0.4139             nan     0.1000    0.0034
##    150        0.3911             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1904
##      2        1.4837             nan     0.1000    0.1392
##      3        1.3938             nan     0.1000    0.1175
##      4        1.3185             nan     0.1000    0.0867
##      5        1.2628             nan     0.1000    0.0892
##      6        1.2048             nan     0.1000    0.0871
##      7        1.1495             nan     0.1000    0.0830
##      8        1.0979             nan     0.1000    0.0620
##      9        1.0586             nan     0.1000    0.0516
##     10        1.0257             nan     0.1000    0.0523
##     20        0.7653             nan     0.1000    0.0305
##     40        0.4778             nan     0.1000    0.0172
##     60        0.3182             nan     0.1000    0.0124
##     80        0.2229             nan     0.1000    0.0057
##    100        0.1608             nan     0.1000    0.0036
##    120        0.1199             nan     0.1000    0.0028
##    140        0.0905             nan     0.1000    0.0022
##    150        0.0791             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2484
##      2        1.4498             nan     0.1000    0.1771
##      3        1.3366             nan     0.1000    0.1530
##      4        1.2392             nan     0.1000    0.1279
##      5        1.1594             nan     0.1000    0.1035
##      6        1.0937             nan     0.1000    0.1049
##      7        1.0285             nan     0.1000    0.0859
##      8        0.9758             nan     0.1000    0.0817
##      9        0.9246             nan     0.1000    0.0676
##     10        0.8822             nan     0.1000    0.0661
##     20        0.5832             nan     0.1000    0.0406
##     40        0.3021             nan     0.1000    0.0155
##     60        0.1694             nan     0.1000    0.0085
##     80        0.1030             nan     0.1000    0.0031
##    100        0.0665             nan     0.1000    0.0009
##    120        0.0474             nan     0.1000    0.0011
##    140        0.0346             nan     0.1000    0.0007
##    150        0.0298             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1312
##      2        1.5206             nan     0.1000    0.0927
##      3        1.4594             nan     0.1000    0.0683
##      4        1.4143             nan     0.1000    0.0587
##      5        1.3762             nan     0.1000    0.0528
##      6        1.3425             nan     0.1000    0.0414
##      7        1.3155             nan     0.1000    0.0461
##      8        1.2877             nan     0.1000    0.0427
##      9        1.2604             nan     0.1000    0.0376
##     10        1.2347             nan     0.1000    0.0342
##     20        1.0518             nan     0.1000    0.0203
##     40        0.8336             nan     0.1000    0.0127
##     60        0.6945             nan     0.1000    0.0087
##     80        0.5937             nan     0.1000    0.0052
##    100        0.5174             nan     0.1000    0.0038
##    120        0.4554             nan     0.1000    0.0039
##    140        0.4058             nan     0.1000    0.0034
##    150        0.3829             nan     0.1000    0.0031
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2007
##      2        1.4789             nan     0.1000    0.1410
##      3        1.3870             nan     0.1000    0.1114
##      4        1.3141             nan     0.1000    0.0989
##      5        1.2505             nan     0.1000    0.0790
##      6        1.1986             nan     0.1000    0.0806
##      7        1.1481             nan     0.1000    0.0678
##      8        1.1048             nan     0.1000    0.0673
##      9        1.0631             nan     0.1000    0.0775
##     10        1.0158             nan     0.1000    0.0556
##     20        0.7606             nan     0.1000    0.0338
##     40        0.4614             nan     0.1000    0.0148
##     60        0.3108             nan     0.1000    0.0083
##     80        0.2149             nan     0.1000    0.0053
##    100        0.1559             nan     0.1000    0.0040
##    120        0.1178             nan     0.1000    0.0030
##    140        0.0904             nan     0.1000    0.0018
##    150        0.0797             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2495
##      2        1.4477             nan     0.1000    0.1989
##      3        1.3239             nan     0.1000    0.1382
##      4        1.2346             nan     0.1000    0.1238
##      5        1.1556             nan     0.1000    0.1035
##      6        1.0906             nan     0.1000    0.1026
##      7        1.0258             nan     0.1000    0.0872
##      8        0.9718             nan     0.1000    0.0863
##      9        0.9189             nan     0.1000    0.0638
##     10        0.8788             nan     0.1000    0.0711
##     20        0.5672             nan     0.1000    0.0349
##     40        0.2950             nan     0.1000    0.0127
##     60        0.1704             nan     0.1000    0.0060
##     80        0.1065             nan     0.1000    0.0033
##    100        0.0703             nan     0.1000    0.0024
##    120        0.0492             nan     0.1000    0.0019
##    140        0.0359             nan     0.1000    0.0005
##    150        0.0307             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1336
##      2        1.5213             nan     0.1000    0.0896
##      3        1.4614             nan     0.1000    0.0680
##      4        1.4165             nan     0.1000    0.0573
##      5        1.3795             nan     0.1000    0.0532
##      6        1.3457             nan     0.1000    0.0500
##      7        1.3149             nan     0.1000    0.0415
##      8        1.2881             nan     0.1000    0.0376
##      9        1.2623             nan     0.1000    0.0334
##     10        1.2406             nan     0.1000    0.0403
##     20        1.0517             nan     0.1000    0.0235
##     40        0.8349             nan     0.1000    0.0126
##     60        0.6980             nan     0.1000    0.0086
##     80        0.5954             nan     0.1000    0.0070
##    100        0.5176             nan     0.1000    0.0047
##    120        0.4545             nan     0.1000    0.0036
##    140        0.4039             nan     0.1000    0.0033
##    150        0.3814             nan     0.1000    0.0028
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1976
##      2        1.4822             nan     0.1000    0.1452
##      3        1.3881             nan     0.1000    0.1094
##      4        1.3173             nan     0.1000    0.1026
##      5        1.2532             nan     0.1000    0.0898
##      6        1.1965             nan     0.1000    0.0983
##      7        1.1356             nan     0.1000    0.0765
##      8        1.0887             nan     0.1000    0.0610
##      9        1.0496             nan     0.1000    0.0589
##     10        1.0130             nan     0.1000    0.0567
##     20        0.7587             nan     0.1000    0.0247
##     40        0.4639             nan     0.1000    0.0140
##     60        0.3096             nan     0.1000    0.0079
##     80        0.2154             nan     0.1000    0.0044
##    100        0.1570             nan     0.1000    0.0033
##    120        0.1143             nan     0.1000    0.0021
##    140        0.0859             nan     0.1000    0.0018
##    150        0.0749             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2525
##      2        1.4463             nan     0.1000    0.1871
##      3        1.3269             nan     0.1000    0.1487
##      4        1.2340             nan     0.1000    0.1149
##      5        1.1593             nan     0.1000    0.1179
##      6        1.0877             nan     0.1000    0.0988
##      7        1.0263             nan     0.1000    0.0728
##      8        0.9801             nan     0.1000    0.0865
##      9        0.9267             nan     0.1000    0.0794
##     10        0.8780             nan     0.1000    0.0607
##     20        0.5640             nan     0.1000    0.0338
##     40        0.2882             nan     0.1000    0.0097
##     60        0.1686             nan     0.1000    0.0083
##     80        0.1032             nan     0.1000    0.0043
##    100        0.0664             nan     0.1000    0.0022
##    120        0.0456             nan     0.1000    0.0009
##    140        0.0330             nan     0.1000    0.0008
##    150        0.0280             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1309
##      2        1.5206             nan     0.1000    0.0916
##      3        1.4606             nan     0.1000    0.0668
##      4        1.4163             nan     0.1000    0.0583
##      5        1.3784             nan     0.1000    0.0504
##      6        1.3454             nan     0.1000    0.0484
##      7        1.3150             nan     0.1000    0.0474
##      8        1.2863             nan     0.1000    0.0398
##      9        1.2596             nan     0.1000    0.0356
##     10        1.2372             nan     0.1000    0.0341
##     20        1.0531             nan     0.1000    0.0232
##     40        0.8365             nan     0.1000    0.0134
##     60        0.6987             nan     0.1000    0.0093
##     80        0.6007             nan     0.1000    0.0060
##    100        0.5218             nan     0.1000    0.0028
##    120        0.4585             nan     0.1000    0.0036
##    140        0.4089             nan     0.1000    0.0027
##    150        0.3869             nan     0.1000    0.0026
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1990
##      2        1.4803             nan     0.1000    0.1418
##      3        1.3889             nan     0.1000    0.1106
##      4        1.3173             nan     0.1000    0.0982
##      5        1.2540             nan     0.1000    0.0803
##      6        1.2018             nan     0.1000    0.0884
##      7        1.1480             nan     0.1000    0.0842
##      8        1.0966             nan     0.1000    0.0692
##      9        1.0538             nan     0.1000    0.0649
##     10        1.0135             nan     0.1000    0.0523
##     20        0.7559             nan     0.1000    0.0323
##     40        0.4718             nan     0.1000    0.0172
##     60        0.3134             nan     0.1000    0.0090
##     80        0.2233             nan     0.1000    0.0062
##    100        0.1615             nan     0.1000    0.0029
##    120        0.1231             nan     0.1000    0.0031
##    140        0.0945             nan     0.1000    0.0017
##    150        0.0820             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2535
##      2        1.4469             nan     0.1000    0.1935
##      3        1.3249             nan     0.1000    0.1380
##      4        1.2375             nan     0.1000    0.1281
##      5        1.1580             nan     0.1000    0.1110
##      6        1.0897             nan     0.1000    0.0955
##      7        1.0306             nan     0.1000    0.0838
##      8        0.9772             nan     0.1000    0.0850
##      9        0.9242             nan     0.1000    0.0651
##     10        0.8830             nan     0.1000    0.0623
##     20        0.5746             nan     0.1000    0.0297
##     40        0.2979             nan     0.1000    0.0135
##     60        0.1731             nan     0.1000    0.0067
##     80        0.1042             nan     0.1000    0.0045
##    100        0.0697             nan     0.1000    0.0020
##    120        0.0500             nan     0.1000    0.0013
##    140        0.0360             nan     0.1000    0.0005
##    150        0.0314             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1366
##      2        1.5200             nan     0.1000    0.0902
##      3        1.4599             nan     0.1000    0.0690
##      4        1.4143             nan     0.1000    0.0606
##      5        1.3740             nan     0.1000    0.0522
##      6        1.3411             nan     0.1000    0.0469
##      7        1.3111             nan     0.1000    0.0424
##      8        1.2841             nan     0.1000    0.0390
##      9        1.2588             nan     0.1000    0.0397
##     10        1.2324             nan     0.1000    0.0338
##     20        1.0468             nan     0.1000    0.0252
##     40        0.8258             nan     0.1000    0.0112
##     60        0.6899             nan     0.1000    0.0078
##     80        0.5925             nan     0.1000    0.0068
##    100        0.5131             nan     0.1000    0.0054
##    120        0.4487             nan     0.1000    0.0036
##    140        0.3981             nan     0.1000    0.0039
##    150        0.3766             nan     0.1000    0.0033
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1963
##      2        1.4798             nan     0.1000    0.1500
##      3        1.3850             nan     0.1000    0.1075
##      4        1.3153             nan     0.1000    0.0969
##      5        1.2522             nan     0.1000    0.0933
##      6        1.1936             nan     0.1000    0.0722
##      7        1.1470             nan     0.1000    0.0834
##      8        1.0955             nan     0.1000    0.0637
##      9        1.0543             nan     0.1000    0.0640
##     10        1.0162             nan     0.1000    0.0556
##     20        0.7606             nan     0.1000    0.0288
##     40        0.4647             nan     0.1000    0.0132
##     60        0.3073             nan     0.1000    0.0084
##     80        0.2173             nan     0.1000    0.0064
##    100        0.1567             nan     0.1000    0.0037
##    120        0.1157             nan     0.1000    0.0029
##    140        0.0884             nan     0.1000    0.0021
##    150        0.0777             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2540
##      2        1.4441             nan     0.1000    0.1751
##      3        1.3317             nan     0.1000    0.1463
##      4        1.2389             nan     0.1000    0.1266
##      5        1.1594             nan     0.1000    0.1077
##      6        1.0920             nan     0.1000    0.0990
##      7        1.0285             nan     0.1000    0.0964
##      8        0.9695             nan     0.1000    0.0832
##      9        0.9180             nan     0.1000    0.0621
##     10        0.8798             nan     0.1000    0.0590
##     20        0.5729             nan     0.1000    0.0354
##     40        0.2877             nan     0.1000    0.0174
##     60        0.1622             nan     0.1000    0.0088
##     80        0.1004             nan     0.1000    0.0025
##    100        0.0657             nan     0.1000    0.0019
##    120        0.0465             nan     0.1000    0.0009
##    140        0.0341             nan     0.1000    0.0008
##    150        0.0297             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1327
##      2        1.5204             nan     0.1000    0.0897
##      3        1.4594             nan     0.1000    0.0706
##      4        1.4126             nan     0.1000    0.0541
##      5        1.3763             nan     0.1000    0.0532
##      6        1.3419             nan     0.1000    0.0436
##      7        1.3142             nan     0.1000    0.0441
##      8        1.2864             nan     0.1000    0.0450
##      9        1.2577             nan     0.1000    0.0337
##     10        1.2359             nan     0.1000    0.0317
##     20        1.0493             nan     0.1000    0.0204
##     40        0.8306             nan     0.1000    0.0103
##     60        0.6989             nan     0.1000    0.0105
##     80        0.5961             nan     0.1000    0.0063
##    100        0.5187             nan     0.1000    0.0050
##    120        0.4558             nan     0.1000    0.0035
##    140        0.4042             nan     0.1000    0.0024
##    150        0.3818             nan     0.1000    0.0025
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1998
##      2        1.4817             nan     0.1000    0.1392
##      3        1.3924             nan     0.1000    0.1165
##      4        1.3177             nan     0.1000    0.0944
##      5        1.2554             nan     0.1000    0.0816
##      6        1.2034             nan     0.1000    0.0865
##      7        1.1500             nan     0.1000    0.0837
##      8        1.0989             nan     0.1000    0.0717
##      9        1.0540             nan     0.1000    0.0681
##     10        1.0122             nan     0.1000    0.0524
##     20        0.7555             nan     0.1000    0.0403
##     40        0.4610             nan     0.1000    0.0139
##     60        0.3116             nan     0.1000    0.0133
##     80        0.2164             nan     0.1000    0.0046
##    100        0.1559             nan     0.1000    0.0042
##    120        0.1160             nan     0.1000    0.0018
##    140        0.0870             nan     0.1000    0.0020
##    150        0.0767             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2520
##      2        1.4466             nan     0.1000    0.1786
##      3        1.3324             nan     0.1000    0.1494
##      4        1.2388             nan     0.1000    0.1286
##      5        1.1576             nan     0.1000    0.1202
##      6        1.0822             nan     0.1000    0.0944
##      7        1.0238             nan     0.1000    0.0831
##      8        0.9715             nan     0.1000    0.0710
##      9        0.9276             nan     0.1000    0.0708
##     10        0.8839             nan     0.1000    0.0605
##     20        0.5893             nan     0.1000    0.0435
##     40        0.2996             nan     0.1000    0.0112
##     60        0.1675             nan     0.1000    0.0058
##     80        0.1041             nan     0.1000    0.0030
##    100        0.0695             nan     0.1000    0.0018
##    120        0.0490             nan     0.1000    0.0010
##    140        0.0351             nan     0.1000    0.0007
##    150        0.0302             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1305
##      2        1.5230             nan     0.1000    0.0896
##      3        1.4627             nan     0.1000    0.0679
##      4        1.4182             nan     0.1000    0.0568
##      5        1.3812             nan     0.1000    0.0505
##      6        1.3490             nan     0.1000    0.0464
##      7        1.3192             nan     0.1000    0.0433
##      8        1.2907             nan     0.1000    0.0375
##      9        1.2665             nan     0.1000    0.0346
##     10        1.2440             nan     0.1000    0.0331
##     20        1.0612             nan     0.1000    0.0186
##     40        0.8489             nan     0.1000    0.0119
##     60        0.7112             nan     0.1000    0.0090
##     80        0.6094             nan     0.1000    0.0083
##    100        0.5299             nan     0.1000    0.0050
##    120        0.4672             nan     0.1000    0.0045
##    140        0.4163             nan     0.1000    0.0042
##    150        0.3932             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1901
##      2        1.4856             nan     0.1000    0.1403
##      3        1.3963             nan     0.1000    0.1079
##      4        1.3273             nan     0.1000    0.1036
##      5        1.2609             nan     0.1000    0.0812
##      6        1.2088             nan     0.1000    0.0740
##      7        1.1611             nan     0.1000    0.0747
##      8        1.1134             nan     0.1000    0.0624
##      9        1.0737             nan     0.1000    0.0500
##     10        1.0422             nan     0.1000    0.0555
##     20        0.7877             nan     0.1000    0.0395
##     40        0.4824             nan     0.1000    0.0161
##     60        0.3230             nan     0.1000    0.0100
##     80        0.2285             nan     0.1000    0.0066
##    100        0.1666             nan     0.1000    0.0043
##    120        0.1220             nan     0.1000    0.0023
##    140        0.0913             nan     0.1000    0.0015
##    150        0.0803             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2469
##      2        1.4529             nan     0.1000    0.1904
##      3        1.3329             nan     0.1000    0.1510
##      4        1.2382             nan     0.1000    0.1198
##      5        1.1624             nan     0.1000    0.1052
##      6        1.0961             nan     0.1000    0.0914
##      7        1.0383             nan     0.1000    0.0881
##      8        0.9851             nan     0.1000    0.0886
##      9        0.9321             nan     0.1000    0.0713
##     10        0.8872             nan     0.1000    0.0603
##     20        0.5860             nan     0.1000    0.0390
##     40        0.2955             nan     0.1000    0.0164
##     60        0.1716             nan     0.1000    0.0063
##     80        0.1061             nan     0.1000    0.0042
##    100        0.0695             nan     0.1000    0.0027
##    120        0.0486             nan     0.1000    0.0011
##    140        0.0366             nan     0.1000    0.0003
##    150        0.0320             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1321
##      2        1.5222             nan     0.1000    0.0898
##      3        1.4634             nan     0.1000    0.0693
##      4        1.4177             nan     0.1000    0.0537
##      5        1.3813             nan     0.1000    0.0551
##      6        1.3466             nan     0.1000    0.0455
##      7        1.3170             nan     0.1000    0.0456
##      8        1.2892             nan     0.1000    0.0413
##      9        1.2623             nan     0.1000    0.0394
##     10        1.2353             nan     0.1000    0.0386
##     20        1.0520             nan     0.1000    0.0205
##     40        0.8335             nan     0.1000    0.0112
##     60        0.6992             nan     0.1000    0.0067
##     80        0.6004             nan     0.1000    0.0065
##    100        0.5190             nan     0.1000    0.0060
##    120        0.4573             nan     0.1000    0.0037
##    140        0.4048             nan     0.1000    0.0039
##    150        0.3839             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1962
##      2        1.4837             nan     0.1000    0.1427
##      3        1.3919             nan     0.1000    0.1172
##      4        1.3180             nan     0.1000    0.0942
##      5        1.2576             nan     0.1000    0.0836
##      6        1.2038             nan     0.1000    0.0745
##      7        1.1567             nan     0.1000    0.0734
##      8        1.1102             nan     0.1000    0.0773
##      9        1.0629             nan     0.1000    0.0660
##     10        1.0224             nan     0.1000    0.0604
##     20        0.7589             nan     0.1000    0.0273
##     40        0.4627             nan     0.1000    0.0142
##     60        0.3128             nan     0.1000    0.0101
##     80        0.2187             nan     0.1000    0.0046
##    100        0.1614             nan     0.1000    0.0026
##    120        0.1210             nan     0.1000    0.0018
##    140        0.0936             nan     0.1000    0.0026
##    150        0.0813             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2482
##      2        1.4481             nan     0.1000    0.1882
##      3        1.3293             nan     0.1000    0.1503
##      4        1.2338             nan     0.1000    0.1280
##      5        1.1545             nan     0.1000    0.1119
##      6        1.0840             nan     0.1000    0.0879
##      7        1.0269             nan     0.1000    0.0899
##      8        0.9720             nan     0.1000    0.0878
##      9        0.9178             nan     0.1000    0.0722
##     10        0.8730             nan     0.1000    0.0538
##     20        0.5718             nan     0.1000    0.0340
##     40        0.2980             nan     0.1000    0.0125
##     60        0.1655             nan     0.1000    0.0081
##     80        0.1019             nan     0.1000    0.0043
##    100        0.0666             nan     0.1000    0.0010
##    120        0.0469             nan     0.1000    0.0016
##    140        0.0340             nan     0.1000    0.0010
##    150        0.0291             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1345
##      2        1.5211             nan     0.1000    0.0915
##      3        1.4613             nan     0.1000    0.0725
##      4        1.4145             nan     0.1000    0.0549
##      5        1.3778             nan     0.1000    0.0501
##      6        1.3439             nan     0.1000    0.0468
##      7        1.3144             nan     0.1000    0.0444
##      8        1.2868             nan     0.1000    0.0434
##      9        1.2588             nan     0.1000    0.0391
##     10        1.2342             nan     0.1000    0.0362
##     20        1.0474             nan     0.1000    0.0198
##     40        0.8327             nan     0.1000    0.0119
##     60        0.6984             nan     0.1000    0.0102
##     80        0.5962             nan     0.1000    0.0048
##    100        0.5212             nan     0.1000    0.0054
##    120        0.4577             nan     0.1000    0.0029
##    140        0.4063             nan     0.1000    0.0039
##    150        0.3859             nan     0.1000    0.0028
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2017
##      2        1.4780             nan     0.1000    0.1396
##      3        1.3868             nan     0.1000    0.1188
##      4        1.3119             nan     0.1000    0.0975
##      5        1.2502             nan     0.1000    0.0832
##      6        1.1954             nan     0.1000    0.0823
##      7        1.1434             nan     0.1000    0.0642
##      8        1.1027             nan     0.1000    0.0713
##      9        1.0590             nan     0.1000    0.0520
##     10        1.0261             nan     0.1000    0.0598
##     20        0.7495             nan     0.1000    0.0269
##     40        0.4693             nan     0.1000    0.0178
##     60        0.3127             nan     0.1000    0.0100
##     80        0.2167             nan     0.1000    0.0063
##    100        0.1565             nan     0.1000    0.0034
##    120        0.1158             nan     0.1000    0.0030
##    140        0.0872             nan     0.1000    0.0021
##    150        0.0769             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2537
##      2        1.4453             nan     0.1000    0.1837
##      3        1.3293             nan     0.1000    0.1469
##      4        1.2362             nan     0.1000    0.1253
##      5        1.1569             nan     0.1000    0.1036
##      6        1.0915             nan     0.1000    0.1050
##      7        1.0264             nan     0.1000    0.0837
##      8        0.9740             nan     0.1000    0.0882
##      9        0.9199             nan     0.1000    0.0709
##     10        0.8770             nan     0.1000    0.0675
##     20        0.5703             nan     0.1000    0.0391
##     40        0.2965             nan     0.1000    0.0115
##     60        0.1679             nan     0.1000    0.0076
##     80        0.1037             nan     0.1000    0.0037
##    100        0.0693             nan     0.1000    0.0011
##    120        0.0489             nan     0.1000    0.0010
##    140        0.0362             nan     0.1000    0.0006
##    150        0.0314             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2447
##      2        1.4517             nan     0.1000    0.1844
##      3        1.3344             nan     0.1000    0.1472
##      4        1.2412             nan     0.1000    0.1174
##      5        1.1648             nan     0.1000    0.1183
##      6        1.0916             nan     0.1000    0.0894
##      7        1.0346             nan     0.1000    0.1013
##      8        0.9730             nan     0.1000    0.0736
##      9        0.9274             nan     0.1000    0.0676
##     10        0.8848             nan     0.1000    0.0638
##     20        0.5730             nan     0.1000    0.0368
##     40        0.2882             nan     0.1000    0.0130
##     60        0.1652             nan     0.1000    0.0083
##     80        0.1023             nan     0.1000    0.0041
##    100        0.0679             nan     0.1000    0.0021
##    120        0.0479             nan     0.1000    0.0010
##    140        0.0354             nan     0.1000    0.0009
##    150        0.0309             nan     0.1000    0.0007
```

```r
pred1 <- predict(model1b, newdata=traintest)
cm <- confusionMatrix(pred1, traintest$classe)
print(cm, digits=4)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    2    0    0    0
##          B    0  946    0    0    0
##          C    0    1  848    0    0
##          D    0    0    7  804    1
##          E    0    0    0    0  900
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9978         
##                  95% CI : (0.996, 0.9989)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9972         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9968   0.9918   1.0000   0.9989
## Specificity            0.9994   1.0000   0.9998   0.9980   1.0000
## Pos Pred Value         0.9986   1.0000   0.9988   0.9901   1.0000
## Neg Pred Value         1.0000   0.9992   0.9983   1.0000   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1929   0.1729   0.1639   0.1835
## Detection Prevalence   0.2849   0.1929   0.1731   0.1656   0.1835
## Balanced Accuracy      0.9997   0.9984   0.9958   0.9990   0.9994
```

The out of sample error is 1-0.9947 = 0.0053.

It was intended to also test with a random forest method, but time did not permit same for the writeup.  Further the initial run resulted in an accuracy of 0.9984 - only moderately better than the boosted tree model, and hence deemed not worth the extra processing time.

Now applying the model to test data set.

## Prepare Final Model


```r
set.seed(72719)
finalPred <- predict(model1b, newdata=testdata)
```

## Applying code for submission


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(finalPred)
```
