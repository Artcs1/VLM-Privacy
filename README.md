<div align="center">
<h2> [TBD] Beyond Blanket Masking: Examining Granularity for Privacy
Protection in Images Captured by Blind and Low Vision Users </h2>

Jeffri Murrugarra-Llerena<sup>1</sup>, Haoran Niu<sup>2</sup>, K. Suzanne Barber, Hal Daumé III<sup>3</sup>, Yang Trista Cao<sup>2</sup>, Paola Cascante-Bonilla<sup>1,</sup><sup>3</sup>

<sup>1</sup> State University of New York at Stony Brook&nbsp; <sup>2</sup>  University of Texas at Austin&nbsp; <sup>3</sup> University of Maryland, College Park

</div>

[[`Paper`]()] [[`Project`](https://github.com/Artcs1/VLM-Privacy)] [[`BibTeX`](#citation)]

## About

## Install

1. Create a conda enviorenment

```
conda create --name py10-vlm python=3.10
conda activate py10-vlm
```
2. Install the following package

```
pip install --upgrade git+https://github.com/huggingface/transformers accelerate
pip install -U flash-attn --no-build-isolation
pip install qwen-vl-utils[decord]
python -m pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```
3. Install the requirements.txt

```
pip install -r requirements.txt
```

## Demo

We show how to use our agents and their interactions in demo.ipynb

## Code for validation experiments

Coming soon ...

## Citation

If you use this model in your research, please consider citing:

```
```
