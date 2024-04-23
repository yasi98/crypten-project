# crypten_vfl_demo



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
