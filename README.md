Hybrid Recommendation System with LightFM
=========================================

A simple hybrid recommendation system built using the **LightFM** library, combining collaborative and content-based signals to generate personalized item recommendations. [github](https://github.com/AjNavneet/Hybrid-Recommender-LightFM-Retail)

## Project Overview

This notebook demonstrates how to:
- Load and preprocess interaction, user, and item data for a recommender system.
- Build sparse matrices for interactions, user features, and item features compatible with LightFM.
- Train a hybrid LightFM model using different loss functions (e.g., WARP, BPR, logistic) for implicit feedback. 
- Evaluate model performance with ranking metrics such as precision and recall at k.
- Visualize interactions, feature distributions, and recommendation quality with plots inside the notebook.

## Tech Stack

- Python (Jupyter Notebook)
- LightFM for hybrid recommendation modeling
- SciPy sparse matrices for interaction and feature representation 
- NumPy, pandas for data handling 
- Matplotlib / Seaborn for plots and visualizations 

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
   (Or manually install `lightfm`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`.)
4. Open the notebook:
   ```bash
   jupyter notebook recsys.ipynb
   ```
5. Run all cells from top to bottom to reproduce preprocessing, training, evaluation, and plots.

## Notebook Structure

- Data loading & basic EDA (distributions, sparsity, interaction heatmaps).
- Feature engineering for users and items (building hybrid feature matrices).
- LightFM model training with different loss functions.
- Evaluation (precision@k / recall@k plots, comparison by model and hyperparameters).

## Intended Use

This notebook is intended as a starting point for:
- Experimenting with **hybrid** recommenders on your own datasets using LightFM.
- Comparing ranking losses and hyperparameters for recommendation quality.
- Currently will have to work more on feature engineering depending on the REAL DATA. This is purely based on dummy data so the outputs aren't our priority, but the plots/visualizations are something we can consider/work with
