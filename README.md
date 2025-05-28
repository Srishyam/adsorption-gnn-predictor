# adsorption-gnn-predictor

# Predicting Adsorption Energies Using Graph Neural Networks

## Summary
This repo implements the CGCNN and ALIGNN models to predict adsorption energies of small molecules (e.g., OH, O, CO, N, OOH) on catalyst surfaces using two open-source datasets (GAS and OCP). This work accompanies the research paper:
### Small-Molecule Adsorption Energy Predictions for High-Throughput Screening of Electrocatalysts
_Srishyam Raghavan, Brian P. Chaplin, and Shafigh Mehraeen_, _Journal of Chemical Information and Modeling_ 2023 63 (17), 5529-5538  
(DOI: https://pubs.acs.org/doi/10.1021/acs.jcim.3c00979) 

## Features
* Compare descriptor-based prediction with ML-based prediction  
* Trains ALIGNN and CGCNN on GAS/OCP datasets  
* Predicts OER overpotential  
* Provides data splits, training scripts, and visualizations

## Datasets
* GASdb (Generalized Adsorption database): https://arxiv.org/abs/1912.10066  
* Open Catalyst Project (OCP) database: https://doi.org/10.48550/arXiv.2010.09990

## Machine Learning models (GNNs)
* Crystal Graph Convolutional Neural Network (CGCNN): https://link.aps.org/doi/10.1103/PhysRevLett.120.145301  
* Atomistic Line Graph Neural Network (ALIGNN): https://doi.org/10.1038/s41524-021-00650-1

