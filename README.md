# Self-Supervised Hypergraph Transformer for Recommender Systems

This repository contains TensorFlow codes and datasets for the paper:
> Lianghao Xia, Chao Huang, Chuxu Zhang (2022). Self-Supervised Hypergraph Transformer for Recommender Systems. In KDD'22, Washington DC, USA, August 14-18, 2022.

## Introduction
This paper proposed a hypergraph transformer model named SHT to tackle the data sparsity and noise problem in collaborative filtering. It combines user/item id embedding with the topology-aware embedding given by GCNs as the local view, and learns the latent global relations in a transformer-like hypergraph encoder. The local embeddings are then denoised by the global view using an augmented solidity differentiation task. Comparing to our previous work <a href='https://github.com/akaxlh/HCCF'>HCCF</a> which is on hypergraph contrastive learning for CF, this paper proposes the novel hypergraph transformer architecture with the self-augmented solidity differentiation task, and highlights the strength of generative self-supervised learning.

## Citation
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{sht2022,
  author    = {Xia, Lianghao and
               Huang, Chao and
	       Zhang, Chuxu},
  title     = {Self-Supervised Hypergraph Transformer for Recommender Systems},
  booktitle = {Proceedings of the 28th {ACM} {SIGKDD} Conference on
               Knowledge Discovery and Data Mining, {KDD} 2022, 
	       Washington DC, USA, August 14-18, 2022.},
  year      = {2022},
}
```

## Environment
The codes of HCCF are implemented and tested under the following development environment:
* python=3.7.0
* tensorflow=1.15.0
* numpy=1.21.6
* scipy=1.7.3

## Datasets
Three datasets are adopted to evaluate SHT: <i> Yelp, Gowalla, </i>and <i>Tmall</i>. The user-item pair $(u_i, v_j)$ in the adjacent matrix is set as 1, if user $u_i$ has rated item $v_j$ in Yelp, or if user $u_i$ has check in venue $v_j$ in Gowalla, or if user $u_i$ has purchased item $v_j$ in Tmall. We filtered out users and items with too few interactions.

## How to Run the Codes
Please unzip the datasets first. Also you need to create the `Models/` directory. The following command lines start training and testing on the three datasets, respectively, which also specify the hyperparameter settings for the reported results in the paper. Training and testing logs for trained models are contained in the `History/` directory.

* Yelp
```
python .\labcode_hop.py --data yelp --reg 1e-2 --ssl_reg 1e-5 --mult 1e2 --edgeSampRate 0.1 --epoch 1
```
* Gowalla
```
python .\labcode_hop.py --data gowalla --reg 1e-2 --ssl_reg 1e-5 --mult 1e1 --epoch 150 --edgeSampRate 0.1 --save_path gowalla_repeat
 ```
 * Tmall
```
python .\labcode_hop.py --data tmall --reg 1e-2 --ssl_reg 1e-5 --mult 1e1 --edgeSampRate 0.1 --epoch 150 --save_path tmall_repeat
```
Important arguments:
* `reg`: This is the weight for weight-decay regularization. Empirically recommended tuning range is `{1e-2, 1e-3, 1e-4, 1e-5}`.
* `ssl_reg`: This is the weight for the solidity prediction loss of self-supervised learning task. The value is tuned from `{1e-3, 1e-4, 1e-5, 1e-6, 1e-7}`.
* `mult`: This hyperparameter is to emplify the ssl loss for better performance, which is tuned from `{16, 64, 1e1, 1e2, 1e3}`.
* `edgeSampRate`: This parameter determines the ratio of edges to conduct the solidity differentiation task on. It should be balanced to consider both model performance and training efficiency.

## Achnowledgements
This research work is supported by the research grants from the Department of Computer Science & Musketeers Foundation Institute of Data Science at the University of Hong Kong.
