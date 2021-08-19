# PyTorch RGCN (Link Prediction)

PyTorch implementation of Relational Link Prediction of RGCN (Modeling Relational Data with Graph Convolutional Networks). The code is sparsely optimized with torch_geometric library, which is builded based on PyTorch.

## About

This is the original tensorflow implementation of link prediction of RGCN:
https://github.com/MichSchli/RelationPrediction

Also, most of the functions in utils.py are brought from the following repository:
https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn,
where this repositroy is based on another popular graph library, namely dgl.


## Requirements
This code is lastly tested with:
* python 3.7.x
* pytorch 1.7.x
* torch_geometric 1.7.x, with torch_scatter 2.0.6 and torch_sparse 0.6.9