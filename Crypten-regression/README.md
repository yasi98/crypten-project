# crypten_vfl_demo

使用crypten做纵向联邦学习demo。详细介绍参考[知乎专栏](https://zhuanlan.zhihu.com/p/364257618)

#Data download

Use the UCI ADULT data set. After downloading the data set, decompress it to obtain the training data adult.data and the test data adult.test. Rename adult.data to adult.train.csv and adult.test to adult.test.csv. Place Go to the project directory

## data preprocessing

Run data_process.py for data preprocessing
```
python data_process.py
```

## Stand-alone training

Run train_single.py for single-machine training
```
python train_single.py
```

## Single-machine vertical federated learning

Suppose there are 3 participants, namely A, B, and C.

Open three terminals and enter commands respectively:

```
RENDEZVOUS=file:///tmp/vfl WORLD_SIZE=3 RANK=0 python train_multi.py
```

```
RENDEZVOUS=file:///tmp/vfl WORLD_SIZE=3 RANK=1 python train_multi.py
```

```
RENDEZVOUS=file:///tmp/vfl WORLD_SIZE=3 RANK=2 python train_multi.py
```

You can start stand-alone vertical federated learning


## Multi-machine vertical federated learning

Suppose there are 3 participants, namely A, B, and C.

A's IP is 192.168.1.100, B's IP is 192.168.1.101, and C's IP is 192.168.1.102. A, B, and C can access each other.
Copy the project files to 3 machines respectively.
Copy the data under dataset/a to A, copy the data under dataset/b to B, and copy the files under dataset/c to C.

In the terminals of the three machines A, B, and C, enter the commands respectively.

A
```
RENDEZVOUS=tcp://192.168.1.100:2345 WORLD_SIZE=3 RANK=0 python train_multi.py
```

B
```
RENDEZVOUS=tcp://192.168.1.100:2345 WORLD_SIZE=3 RANK=1 python train_multi.py
```

C
```
RENDEZVOUS=tcp://192.168.1.100:2345 WORLD_SIZE=3 RANK=2 python train_multi.py
```

You can start stand-alone vertical federated learning

## Comparison of training results

Single machine vs vertical federated learning

| epoch | stand-alone auc| Vertical federated learning auc |
| --- | --- | --- |
| 1 | 0.635 | 0.626 |
| 2 | 0.766 | 0.755 |
| 3 | 0.813 | 0.809 |
| 4 | 0.831 | 0.831 |
| 5 | 0.841 | 0.843 |
| 6 | 0.847 | 0.850 |
| 7 | 0.852 | 0.856 |
| 8 | 0.856 | 0.859 |
| 9 | 0.859 | 0.863 |
| 10 | 0.862 | 0.866 |
| 11 | 0.864 | 0.868 |
| 12 | 0.867 | 0.871 |
| 13 |  0.869 | 0.873 |
| 14 | 0.871 | 0.876 |
| 15 | 0.874 | 0.878 |
| 16 | 0.876 | 0.880 |
| 17 | 0.878 | 0.882 |
| 18 | 0.879 | 0.884 |
| 19 | 0.881 | 0.885 |
| 20 | 0.883 | 0.887 |

