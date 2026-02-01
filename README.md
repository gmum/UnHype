<div align="center">

# UnHype: CLIP-Guided Hypernetworks for Dynamic LoRA Unlearning

Piotr Wójcik*, Maksym Petrenko*, Wojciech Gromski*, Przemysław Spurek, Maciej Zięba <br>
(* denotes equal contribution)

![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b.svg)
[![GitHub stars](https://img.shields.io/github/stars/gmum/UnHype?style=social)](https://github.com/gmum/UnHype)

</div>

![Teaser](assets/teaser.png)

This repository contains the implementation for the paper **UnHype: CLIP-Guided Hypernetworks for Dynamic LoRA Unlearning**.

**Abstract:** Recent advances in large-scale diffusion models have intensified concerns about their potential misuse, particularly in generating realistic yet harmful or socially disruptive content. This challenge has spurred growing interest in effective machine unlearning, the process of selectively removing specific knowledge or concepts from a model without compromising its overall generative capabilities. Among various approaches, Low-Rank Adaptation (LoRA) has emerged as an effective and efficient method for fine-tuning models toward targeted unlearning. However, LoRA-based methods often exhibit limited adaptability to concept semantics and struggle to balance removing closely related concepts with maintaining generalization across broader meanings. Moreover, these methods face scalability challenges when multiple concepts must be erased simultaneously. To address these limitations, we introduce UnHype, a framework that incorporates hypernetworks into single- and multi-concept LoRA training. The proposed architecture can be directly plugged into Stable Diffusion as well as modern flow-based text-to-image models, where it demonstrates stable training behavior and effective concept control. During inference, the hypernetwork dynamically generates adaptive LoRA weights based on the CLIP embedding, enabling more context-aware, scalable unlearning. We evaluate UnHype across several challenging tasks, including object erasure, celebrity erasure, and explicit content removal, demonstrating its effectiveness and versatility.

## Code

The code will be uploaded soon. Stay tuned!

## Citation

If you find this work useful, please consider citing our paper.
