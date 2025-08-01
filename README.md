# Exploiting Asymmetric Image-to-Text Alignment for Black-Box Membership Inference in Large Vision-Language Models

## About The Project
**AlignMIA** is a black-box membership inference attack that exploits semantic consistency under visual perturbations to detect training data in LVLMs.
It applies randomized patch-level masking and captures alignment-driven robustness in textual outputs to construct discriminative semantic trajectories as membership features.
<br>

## Getting Started
### File Structure 
```
ALIGNMIA-CODE
├── llava
├── mgm
├── minigpt4
├── data_process.py
├── eval.py
└── main.py

```
There are several parts of the code:

- `llava`, `mgm`, `minigpt4`: These three folders contain model-specific implementations for the LVLM architectures used in our experiments—LLaVA, MiniGemini, and MiniGPT-4, respectively. Each directory includes essential modules for model initialization, inference routines, and loss computation tailored to the corresponding architecture.
- `data_process.py`: This file contains utilities for data preprocessing and formatting.
- `main.py`: The main function of **AlignMIA**, implementing the core attack procedure, including randomized patch-level masking and membership feature generation.
- `eval.py`: This file handles the training of the anomaly detection model and the evaluation of membership inference performance.
<br>

### Requirements

* python 3.11.10
* [pytorch](https://pytorch.org/get-started/locally/) 2.0.1 & torchvision 0.15.2 
* CUDA 11.8 and above are recommended (this is for GPU users)
* numpy 1.26.4
* scipy 1.14.1
* rouge-score 1.0.1
* nltk 3.8.1
* pyod 2.0.3
* scikit-learn 1.3.2
* transformers 4.49.0
* pillow 10.2.0
* tqdm 4.67.0

Before running the project, make sure you have set up the correct environment and installed all required dependencies. To ensure compatibility with the target LVLMs, you may also need to install additional configuration libraries specific to each model. For details, please refer to the official repositories of the respective models:

LLaVA: https://github.com/haotian-liu/LLaVA

MiniGemini: https://github.com/dvlab-research/MGM

MiniGPT-4: https://github.com/Vision-CAIR/MiniGPT-4
<br>

### Hyper-parameters 
The settings of **AlignMIA** are determined in the parameter **args** in `main.py`. Here, we mainly introduce the important hyper-parameters.
- model_path: path to the target LVLM checkpoint. Default: 'checkpoints/LLaVA-7B-v1.5'.
- patch_size: size of patches used for visual masking. Default: 14.
- random_times: number of randomized masking repetitions per mask ratio. Default: 10.
- rate_list: list of masking ratios applied to the image input. Default: [0.0, 0.25, 0.50, 0.75].
- save_dir: directory prefix for saving model outputs and intermediate results. Default: './model_outputs'.
- gen_len: maximum length of the generated textual response. Default: 50.
- task_type: task format for generation; supports "short" and "long". Default: 'short'.
- train_num: number of auxiliary (non-member) samples used to train the MIA model. Default: 1000.
- train_seed: random seed for shuffling training data. Default: 42.
- val_nonmemnum: number of non-member samples used for validation. Default: 500.
- val_ratio: ratio of member to non-member samples in validation. Default: 1.0.
- val_seed: random seed for shuffling validation data. Default: 42.
- seed: global random seed for experiment reproducibility. Default: 42.
- local_seed: local seed for image masking and sampling. Default: 42.
- image_size: Size of the input images (e.g.336x336). Determines the total number of patches that can be masked. If set to None, the default image size of the corresponding LVLM model will be used.  Default: None.
<br>

### Run
To reproduce our attack pipeline, please execute the following scripts in order:
**1. Preprocess the auxiliary datasets**
```python
$ python data_process.py
```
Since the auxiliary datasets we use come in various formats, we have written custom parsing and formatting functions for each of them. To run this script, you need to manually specify the path to each dataset in the main() function at the end of data_process.py. If you do not wish to use all datasets, feel free to comment out the corresponding parts in the code.
We support the following datasets, and you can download them from their respective official sources:

- Instruction-tuned datasets:

LLaVA: [LLaVA v1.5 mix665k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)

MiniGemini: [MGM Instruction](https://huggingface.co/datasets/YanweiLi/MGM-Instruction)

MiniGPT-4: [CC_SBU_Aign](https://drive.google.com/file/d/1nJXhoEcy3KTExr17I7BXqY5Y9Lx_-n-9/view)

- Auxiliary datasets for short-answer task:

VQAv2: [images](http://images.cocodataset.org/zips/val2014.zip), [questions](https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip), [annotations](https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)

VizWiz-VQA: [images](https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip), [annotations](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip)

ImageNet-1k: [download](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description)

Q-Bench: [images](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/images_llvisionqa.tar), [annotations](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/llvisionqa_dev.json)

POPE: [download](https://github.com/RUCAIBox/POPE)

MME: [download](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)

- Auxiliary datasets for long-answer task:

COCO2017: [download](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)

Flickr8k: [download](https://www.kaggle.com/datasets/adityajn105/flickr8k)

LLaVA-Bench-in-the-Wild: [download](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild)

**2. Extract membership features for all samples**
```python
$ python main.py --model_path "path/to/your/target/model"
```
For LLaVA and MiniGemini, you can directly use the officially released fine-tuned checkpoints. Please specify their paths using the --model_path argument and ensure they are properly loaded (e.g., from Hugging Face or local folders).

For MiniGPT-4, you need to manually fine-tune the model yourself using the official instructions before running this script.

**3. Train the attack model and evaluate the membership inference performance**
```python
$ python eval.py
```

<br>

## Note
- we recommend using GPUs to increase the running speed. 

<br>