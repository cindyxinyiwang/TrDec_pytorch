# TrDec: a tree-based decoder for neural machine translation

source code for paper
> [A Tree-based Decoder for Neural Machine Translation] (https://arxiv.org/abs/1808.09374)
> Xinyi Wang, Hieu Pham, Pengcheng Yin, Graham Neubig

## Requirements

Python 3.6, PyTorch 0.4.1

## Training
We provide the data and scripts to run the experiments for the small orormo dataset
To train the oromo TrDec model:
`./scripts/orm_trdec.py`

## Decode
To decode from the trained TrDec model:
`./scripts/orm_trdec_translate.py`

##
We will update with other scripts used for the paper soon
