# VQ-GNN: A Universal Framework to Scale up Graph Neural Networks using Vector Quantization

Official implementation of our VQ-GNN [paper](https://arxiv.org/abs/2110.14363) (NeurIPS2021).

**TL;DR:** VQ-GNN, a principled universal framework to scale up GNNs using Vector Quantization (VQ) without compromising the performance. In contrast to sampling-based techniques, our approach can effectively preserve all the messages passed to a mini-batch of nodes by learning and updating a small number of quantized reference vectors of global node representations.

## Experiments

To reproduce the experimental results in the paper, install the required packages and use the commands listed below.

### Requirements

* torch >= 1.9.0
* torch-geometric >= 1.7.2
* ogb >= 1.3.1

### Commands
ogbn-arxiv `GCN`
```
cd vq_gnn_v2
python main_node.py --num-D 4 --conv-type GCN --dataset arxiv --num-parts 80 --batch-size 40 --test-batch-size 40 --lr 1e-3 --sampler-type cluster
```

ogbn-arxiv `SAGE-Mean`
```
cd vq_gnn_v2
python main_node.py --num-D 4 --conv-type SAGE --dataset arxiv --num-parts 20 --batch-size 10 --test-batch-size 10 --lr 1e-3 --sampler-type cluster
```

ogbn-arxiv `GAT`
```
cd vq_gnn_v2
python main_node.py --num-D 4 --conv-type GAT --dataset arxiv --num-parts 20 --batch-size 10 --test-batch-size 10 --lr 1e-3 --sampler-type cluster
```

ppi `GCN`
```
cd vq_gnn_v2
python main_node.py --hidden-channels 256  --lr 3e-3 --epochs 5000 --batch-size 30000 --test-batch-size 0 --num-M 4096 --num-D 4  --conv-type GCN --sampler-type node --dataset ppi --skip
```

ppi `SAGE-Mean`
```
cd vq_gnn_v2
python main_node.py --hidden-channels 256  --lr 3e-3 --epochs 5000 --batch-size 30000 --test-batch-size 0 --num-M 4096 --num-D 4  --conv-type SAGE --sampler-type node --dataset ppi --skip
```

ppi `GAT`
```
cd vq_gnn_v2
python main_node.py --hidden-channels 256  --lr 3e-3 --epochs 5000 --batch-size 10000 --test-batch-size 0 --num-M 4096 --num-D 4  --conv-type GAT --sampler-type node --dataset ppi --skip
```

ogbl-collab `GCN`
```
cd vq_gnn_v2
python main_link.py  --lr 3e-3 --epochs 400 --log-steps 1  --batch-size 50000 --test-batch-size 80000 --num-M 1024 --num-D 4  --conv-type GCN --sampler-type cont --walk-length 15 --cont-sliding-window 1 --dataset collab --skip
```

ogbl-collab `SAGE-Mean`
```
cd vq_gnn_v2
python main_link.py  --lr 3e-3 --epochs 400 --log-steps 1  --batch-size 50000 --test-batch-size 80000 --num-M 1024 --num-D 4  --conv-type SAGE --sampler-type cont --walk-length 15 --cont-sliding-window 1 --dataset collab
```

ogbl-collab `GAT`
```
cd vq_gnn_v2
python main_link.py  --lr 3e-3 --epochs 400 --log-steps 1  --batch-size 20000 --test-batch-size 80000 --num-M 1024 --num-D 4  --conv-type GAT --sampler-type cont --walk-length 15 --cont-sliding-window 1 --dataset collab --skip
```

reddit `GCN`
```
cd vq_gnn_v1
python main_node.py --hidden-channels 128 --dropout 0 --lr 1e-3 --epochs 100 --batch-size 10000 --test-batch-size 50000 --num-M 1024 --num-D 4 --grad-scale 1 1 --warm-up --momentum 0.1 --conv-type GCN --dataset reddit --sampler-type cont --walk-length 3 --cont-sliding-window 1  --recovery-flag --bn-flag
```

reddit `SAGE-Mean`
```
cd vq_gnn_v1
python main_node.py --hidden-channels 128 --dropout 0 --lr 1e-3 --epochs 100 --batch-size 6000 --test-batch-size 50000 --num-M 1024 --num-D 4 --grad-scale 1 1 --warm-up --momentum 0.1 --conv-type SAGE --dataset reddit --sampler-type cont --walk-length 3 --cont-sliding-window 1  --recovery-flag --bn-flag 
```

reddit `GAT`
```
cd vq_gnn_v1
python main_node.py --hidden-channels 128 --dropout 0 --lr 1e-3 --epochs 100 --batch-size 2000 --test-batch-size 5000 --num-M 1024 --num-D 4 --grad-scale 1 1 --warm-up --momentum 0.1 --conv-type GAT --dataset reddit --sampler-type cont --walk-length 3 --cont-sliding-window 1  --recovery-flag --bn-flag 
```

flickr `GCN`
```
cd vq_gnn_v1
python main_node.py --hidden-channels 128 --dropout 0 --lr 1e-3 --epochs 100 --batch-size 50000 --test-batch-size 0 --num-M 1024 --num-D 4 --grad-scale 1 1 --warm-up --momentum 0.1 --conv-type GCN --dataset flickr --sampler-type cont --walk-length 5 --cont-sliding-window 1  --recovery-flag --bn-flag 
```

flickr `SAGE-Mean`
```
cd vq_gnn_v1
python main_node.py --hidden-channels 128 --dropout 0 --lr 1e-3 --epochs 100 --batch-size 50000 --test-batch-size 0 --num-M 1024 --num-D 4 --grad-scale 1 1 --warm-up --momentum 0.1 --conv-type SAGE --dataset flickr --sampler-type cont --walk-length 5 --cont-sliding-window 1  --recovery-flag --bn-flag 
```

flickr `GAT`
```
cd vq_gnn_v1
python main_node.py --hidden-channels 128 --dropout 0 --lr 1e-3 --epochs 100 --batch-size 30000 --test-batch-size 0 --num-M 1024 --num-D 4 --grad-scale 1 1 --warm-up --momentum 0.1 --conv-type GAT --dataset flickr --sampler-type cont --walk-length 5 --cont-sliding-window 1  --recovery-flag --bn-flag 
```

## Cite
If you find VQ-GNN useful, please cite our paper.
```
@misc{ding2021vqgnn,
      title={VQ-GNN: A Universal Framework to Scale up Graph Neural Networks using Vector Quantization}, 
      author={Mucong Ding and Kezhi Kong and Jingling Li and Chen Zhu and John P Dickerson and Furong Huang and Tom Goldstein},
      year={2021},
      eprint={2110.14363},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

