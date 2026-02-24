Hybrid Recommendation System with LightFM
=========================================

A simple hybrid recommendation system built using the **LightFM** library, combining collaborative and content-based signals to generate personalized item recommendations. [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)

## Project Overview

This notebook demonstrates how to:
- Load and preprocess interaction, user, and item data for a recommender system. [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)
- Build sparse matrices for interactions, user features, and item features compatible with LightFM. [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)
- Train a hybrid LightFM model using different loss functions (e.g., WARP, BPR, logistic) for implicit feedback. [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)
- Evaluate model performance with ranking metrics such as precision and recall at k. [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)
- Visualize interactions, feature distributions, and recommendation quality with plots inside the notebook. [kaggle](https://www.kaggle.com/code/gpreda/hybrid-recsys-evaluation)

## Tech Stack

- Python (Jupyter Notebook) [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)
- LightFM for hybrid recommendation modeling [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)
- SciPy sparse matrices for interaction and feature representation [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)
- NumPy, pandas for data handling [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)
- Matplotlib / Seaborn for plots and visualizations [kaggle](https://www.kaggle.com/code/gpreda/hybrid-recsys-evaluation)

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/AyushDubeyaquagaming/Hybrid-RecSys.git
   cd Hybrid-RecSys/lightfm
   ```
2. Create and activate a virtual environment (optional but recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Or manually install `lightfm`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`.) [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)
4. Open the notebook:
   ```bash
   jupyter notebook recsys.ipynb
   ```
5. Run all cells from top to bottom to reproduce preprocessing, training, evaluation, and plots. [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)

## Notebook Structure

- Data loading & basic EDA (distributions, sparsity, interaction heatmaps). [kaggle](https://www.kaggle.com/code/gpreda/hybrid-recsys-evaluation)
- Feature engineering for users and items (building hybrid feature matrices). [projectpro](https://www.projectpro.io/project-use-case/hybrid-recommender-systems-python-lightfm)
- LightFM model training with different loss functions. [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)
- Evaluation (precision@k / recall@k plots, comparison by model and hyperparameters). [kaggle](https://www.kaggle.com/code/gpreda/hybrid-recsys-evaluation)

## Intended Use

This notebook is intended as a starting point for:
- Experimenting with **hybrid** recommenders on your own datasets using LightFM. [projectpro](https://www.projectpro.io/project-use-case/hybrid-recommender-systems-python-lightfm)
- Comparing ranking losses and hyperparameters for recommendation quality. [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)
- Extending the pipeline into a production-ready recommendation service.
