# Patent Sentence Classification

This repository contains the code developed for the analyses presented in the paper: "Comparing Encoder and Decoder LLMs for New Product Development on Supervised Tasks."

The study was presented at the Conference "London Text Analysis Conference LTAC 2025, Generative AI in Management, Education and Research, 11-12 Settembre, Londra, UK" (https://sites.google.com/view/ltac-2025/home)

The paper is currently under review for publication in a peer-reviewed journal.

## Paper Abstract
The emergence of Large Language Models (LLMs) offers new opportunities for analyzing unstructured textual data, which firms increasingly exploit to support innovation and New Product Development (NPD) activities. However, researchers and practitioners in this domain lack systematic comparative analyses of the different types of LLMs available and their implementation trade-offs. This study compares encoder-only and decoder-only LLMs in terms of classification performance, training data requirements, and computational time. We conduct an experimental evaluation on the identification of functional requirements and design parameters based on axiomatic design theory, considering two supervised tasks: sentence classification and named entity recognition applied to patent texts. While decoder-only LLMs using prompting are easy to deploy, require no annotated data for fine-tuning, and are readily adaptable to new tasks, they consistently underperform encoder-only models, even when only limited annotated training data are available. Our findings indicate that smaller encoder-only models offer more resource-efficient alternatives for real-world NPD applications, and we discuss the conditions under which the use of generative decoder-only models for supervised tasks may constitute a form of methodological misuse. 

Fig. 1 shows _______________________

![comparison_sentence_classification](assets/comparison_sentence_classification.png)

**Fig. 1** - *_________________________*
