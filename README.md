# RiskStratificationPreeclampsia
Risk Stratification for Improved Preeclampsia Prediction by Leveraging Unsupervised Machine Learning:

The following repository contains code I wrote for a research project completed with Dr Brian Cox during the Summer of 2023. Unfortunately, at this time, no outputs are availible as the data is confidential and I have not had a chance to simulate data and rerun the workflow.

The workflow is to be run in the following way:

1. Create virtual environments pappas_tadam and NBclust_env from their requirements.txt files
2. Run the workflow in the following order (you can actually run in any order as the folder provided will have all data files already generated, but the workflow runs in this order if you only have the base data files). Make sure to specify home_dir at the start of each file.

    1. DataPrep.ipynb 
    2. Visualization.ipynb
    3. MatchedDataSetPrep.ipynb
    4. MatchedClustering.ipynb
    5. ClusterEnsemble.ipynb
    6. NBclust.ipynb
    7. ClusterEvaluation.ipynb

Reference Notebooks:

* FunctionsVignette.ipynb
* ClusteringTutorial.ipynb

Functions:

* FunctionsOOPGood.py

More Details about workflow:

https://docs.google.com/document/d/1InNNEsJdqRrRAW7w2ZTVKEYoEtDRDpvS7EQUzkOcu7A/edit?usp=sharing
