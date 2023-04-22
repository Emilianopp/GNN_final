# GNN Final Project

This project is based on predicting communities on temporal dynamic networks. I trained temporal graph attention networks on generated synthetic datasets as well as a subset of the DBLP co-authorship network. 

To generate the synthetic data: 

```
python make_synthetic.py
```


To create the subset of the DBLP network, first download the v-14 version of the DBLP dataset from [here](https://www.aminer.org/citation) 

To generete place the data in the root directory and execute

```
python make_dblp.py --datset_dir ./data/
```


I would like to thank the authors of 

[Towards Better Dynamic Graph Learning New Architecture and Unified Library](https://arxiv.org/pdf/2303.13047.pdf)

for making their [code](https://github.com/yule-BUAA/DyGLib) public as it was it easy to modify to suit my datasets
