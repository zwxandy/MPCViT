# MPCViT
This is the source code of [MPCViT: Searching for MPC-friendly Vision Transformer with Heterogeneous Attention](https://arxiv.org/pdf/2211.13955.pdf).

## Abstract
Secure multi-party computation (MPC) enables computation directly on encrypted data on non-colluding untrusted servers and protects both data and model privacy in deep learning inference. However, existing neural network (NN) architectures, including Vision Transformers (ViTs), are not designed or optimized for MPC protocols and incur significant latency overhead due to the Softmax function in the multi-head attention (MHA). In this paper, we propose an MPC-friendly ViT, dubbed MPCViT, to enable accurate yet efficient ViT inference in MPC. We systematically compare different attention variants in MPC and propose a heterogeneous attention search space, which combines the high-accuracy and MPC-efficient attentions with diverse structure granularities. We further propose a simple yet effective differentiable neural architecture search (NAS) algorithm for fast ViT optimization. MPCViT significantly outperforms prior-art ViT variants in MPC. With the proposed NAS algorithm, our extensive experiments demonstrate that MPCViT achieves 7.9× and 2.8× latency reduction with better accuracy compared to Linformer and MPCFormer on the Tiny-ImageNet dataset, respectively. Further, with proper knowledge distillation (KD), MPCViT even achieves 1.9% better accuracy compared to the baseline ViT with 9.9× latency reduction on the Tiny-ImageNet dataset.

## Training Guidance
Here, we take CIFAR-10 dataset as our training example, and give a brief guidance of how to search and retrain MPCViT. Detailed settings and hyper-parameters for three datasets are listed in both paper and code. Check it if needed!

### Search for the optimal heterogeneous ViT
First, we leverage the idea of differentiable architecture search (DARTS) to search for the optimal heterogeneous ViT, which combines the advantages of both communication-efficent and high-accuracy attention mechanisms.
```shell
python train.py --config configs/datasets/cifar10.yml --model vit_7_4_32 /path/to/cifar-10 --search-mode --epochs 300
```

### Retrain the heterogeneous ViT 
After searching, we obtain the trained architecture parameters. We structurally select a part of attention as RSAttn and the other part as ScalAttn according to the RSAttn ratio.
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