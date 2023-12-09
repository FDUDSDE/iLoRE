# iLoRE
Codes for "iLoRE: Dynamic Graph Representation Learning with Instant Long-term Modeling and Re-occurrence Preservation"  [paper](https://arxiv.org/pdf/2309.02012.pdf)

# Data

Please download data from the [here](https://snap.stanford.edu/jodie/) and pre-process them with the script provided by [TGN](https://github.com/twitter-research/tgn).

# How to use

For the temporal link prediction task, for example, please run:

```
python train_self_supervised.py --data [DATA] --bs [batch_size] --m_bs [mini_batch_size] --count_dim [xx] --block_number [2,3...]
```

For the node classification task, for example, please run:
```
python train_supervised.py --data [DATA] --bs [batch_size] --m_bs [mini_batch_size] --count_dim [xx] --block_number [2,3...]
```
You should run the temporal link prediction task first.

# Cite

```
@inproceedings{zhang2023iLoRE,
  title={iLoRE: Dynamic Graph Representation Learning with Instant Long-term Modeling and Re-occurrence Preservation},
  author={Siwei, Zhang and Yun, Xiong and Yao, Zhang and Xixi, Wu and Yiheng, Sun and Jiawei, Zhang},
  booktitle={The 32nd ACM International Conference on Information and Knowledge Management},
  year={2023}
}

```
