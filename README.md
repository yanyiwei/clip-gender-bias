# Contrastive Language-Vision AI Models Pretrained on Web-Scraped Multimodal Data Exhibit Sexual Objectification Bias

This repo contains code for the paper [Contrastive Language-Vision AI Models Pretrained on Web-Scraped Multimodal Data Exhibit Sexual Objectification Bias](https://arxiv.org/abs/2212.11261) by Robert Wolfe, Yiwei Yang, Bill Howe, Aylin Caliskan.

The SOBEM dataset can be obtained [here](https://sites.google.com/g.unitn.it/smablab/sobem-database).

The code for running Embedding Association Tests (EAT) is in the folders `Embedding Collection`, `Emotion Associations`, and `Profession Associations`. `Embedding Collection` contains code for computing embeddings for the datasets: OASIS, SOBEM, and professional images. `Emotion Associations` and `Profession Associations` contain code for computing the association effect sizes and p values. 

For example, to run EAT on images of professionals, run `python all_sex_profession_collection.py` in directory `Embedding Collection`, then run `python all_sex_profession_associations.py` in directory `Profession Associations`. 

The GradCAM saliency maps can be obtained by executing the jupyter notebook `CLIP_GradCAM_Visualization.ipynb`. 