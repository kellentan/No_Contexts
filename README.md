# No_Contexts
This repository contains the code and implementation for our NAACL'24 paper "No Context Needed: Contextual Quandary in Idiomatic Reasoning with Pre-Trained Language Models".

## Environment Setup
Note that our implementation used conda to setup our environment. The ```env``` folder in this repository contains both a .yml and a .txt environment file. To setup the environment with conda (preferred), please take the ```env/environment.yml``` file and run the following command, replacing ```<env>``` with your own environment name:

```
conda env create --name <env> --file=environment.yml
```

We have also included the ```env/requirements.txt``` file if you would like to run the code via a virtual environment instead. To install the packages that we used, please run the following command within your pip virtual environment:

```
pip install -r requirements.txt
```

## Implementation
For a cleaner (revised) version of the code, please refer to the ```src/context_experiments.ipynb``` notebook, which shows how to run a simple experiment on the IMPLI dataset. You can choose to toggle whether to remove context, shuffle the context, or keep the original samples for evaluation. The methods can be found in the corresponding utility file ```src/context_utils.py```. Note that in this version, only experiments with the IMPLI dataset have been tested (i.e. you may encounter bugs when utilizing the utility methods for the FigurativeNarrativeBenchmark dataset). 

Please refer to the ```src/raw_code.ipynb``` notebook file if you want the complete and original implementation of all of our experiments. Note that regardless of the implementation you run, you will need to replace directory paths for models, tokenizers, and datasets to point to point to the correct locations.

Our code was ran on the Della high performance computing cluster at Princeton University, with GPU access to a single Nvidia A100 (80 GB of VRAM memory). If you need to run the code without GPU access, please set the ```TrainingArguments``` parameter ```fp16=False```. If running on the Della server, our notebook code can be run using a Jupyter session, and declaring the following slurm commands (for GPU access):

```
--gres=gpu:1 --constraint=gpu80
```

## Citation
If you found our paper or this repository useful for your research, please consider citing our paper:
```
@inproceedings{cheng-bhat-2024-context,
    title = "No Context Needed: Contextual Quandary In Idiomatic Reasoning With Pre-Trained Language Models",
    author = "Cheng, Kellen  and
      Bhat, Suma",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.272",
    doi = "10.18653/v1/2024.naacl-long.272",
    pages = "4863--4880",
    abstract = "Reasoning in the presence of idiomatic expressions (IEs) remains a challenging frontier in natural language understanding (NLU). Unlike standard text, the non-compositional nature of an IE makes it difficult for model comprehension, as their figurative or non-literal mean- ing usually cannot be inferred from the constituent words alone. It stands to reason that in these challenging circumstances, pre-trained language models (PTLMs) should make use of the surrounding context to infer additional in- formation about the IE. In this paper, we investigate the utilization of said context for idiomatic reasoning tasks, which is under-explored relative to arithmetic or commonsense reason- ing (Liu et al., 2022; Yu et al., 2023). Preliminary findings point to a surprising observation: general purpose PTLMs are actually negatively affected by the context, as performance almost always increases with its removal. In these scenarios, models may see gains of up to 3.89{\%}. As a result, we argue that only IE-aware models remain suitable for idiomatic reasoning tasks, given the unexpected and unexplainable manner in which general purpose PTLMs reason over IEs. Additionally, we conduct studies to examine how models utilize the context in various situations, as well as an in-depth analysis on dataset formation and quality. Finally, we provide some explanations and insights into the reasoning process itself based on our results.",
}
```
