# MPCViT
This is the source code of [MPCViT: Searching for MPC-friendly Vision Transformer with Heterogeneous Attention](https://arxiv.org/pdf/2211.13955.pdf).

## Training Guidance
Here, we take CIFAR-10 dataset as our training example, and give a brief guidance of how to search and retrain MPCViT. Detailed settings and hyper-parameters for three datasets are listed in both paper and code. Check it if needed!

### Search for the optimal architecture parameter
```shell
python train.py --config configs/datasets/cifar10.yml --model vit_7_4_32 /path/to/cifar-10 --search-mode --epochs 300
```

### Retrain the heterogenuous ViT 
```shell
python train.py --config configs/datasets/cifar10.yml --model vit_7_4_32 /path/to/cifar-10 --retrain-mode --epochs 300 --search-checkpoint output/train/path/model_best.pth.tar --rs-ratio 0.7
```
If you want to train heterogeneous ViT with Knowledge Distillation (KD):
```shell
python train.py --config configs/datasets/cifar10.yml --model vit_7_4_32 /path/to/cifar-10 --retrain-mode --epochs 300 --search-checkpoint output/train/path/model_best.pth.tar --rs-ratio 0.7 --use-kd --teacher-checkpoint /path/to/SoftmaxAttn/model_best.pth.tar
```
Note that we can modify `use-kd` to `use-token-kd` to perform a more fine-grained KD with relatively higher accuracy.

### Inference of loaded ViT checkpoint (a tool, not necessary)
```shell
python inference.py --config configs/datasets/cifar10.yml --model vit_7_4_32 /path/to/cifar-10 --model-checkpoint output/train/path/model_best.pth.tar
```

## Citation
We follow the codebase named 'Compact Transforer' below:
```bibtex
@article{hassani2021escaping,
	title        = {Escaping the Big Data Paradigm with Compact Transformers},
	author       = {Ali Hassani and Steven Walton and Nikhil Shah and Abulikemu Abuduweili and Jiachen Li and Humphrey Shi},
	year         = 2021,
	url          = {https://arxiv.org/abs/2104.05704},
	eprint       = {2104.05704},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}
```