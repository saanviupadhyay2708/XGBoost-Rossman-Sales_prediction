# Rossmann Store Sales Prediction

This project uses gradient boosting with XGBoost to predict daily sales for Rossmann drug stores.

## Project Overview

The goal of this project is to predict daily sales for Rossmann stores based on various features like promotions, competition, holidays, and seasonality. The ability to accurately forecast sales helps store managers plan staff schedules, inventory, and marketing campaigns effectively.

## Dataset

The data comes from the [Kaggle Rossmann Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales) and contains historical sales data for 1,115 Rossmann stores. The dataset includes:

- `train.csv`: Historical sales data including Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, and SchoolHoliday
- `store.csv`: Supplementary information about each store, including StoreType, Assortment, CompetitionDistance, etc.
- `test.csv`: The test set with the same features as the training set (except for Sales and Customers)
- `sample_submission.csv`: A sample submission file in the correct format

## Data Preprocessing

The preprocessing pipeline includes:

1. Merging the training data with store information
2. Converting dates and extracting features (Year, Month, Day, WeekOfYear)
3. Handling closed stores (removing them from training data)
4. Creating competition features:
   - Calculating months since competition started (`CompetitionOpen`)
5. Creating promotion features:
   - Calculating months since Promo2 started (`Promo2Open`) 
   - Flag for whether current month is in the promotion interval (`IsPromo2Month`)
6. Handling missing values in CompetitionDistance
7. Scaling numeric features to 0-1 range
8. One-hot encoding categorical variables

## Feature Engineering

The most important features generated and used:

- Store and promotion information
- Calendar features (Day, Month, Year, WeekOfYear)
- Competition details (distance and duration)
- Promotional campaign details
- Store type and assortment information

## Model Building

The model uses XGBoost regressor with:
- K-fold cross-validation (5 folds)
- Hyperparameter tuning for:
  - Number of estimators (trees)
  - Maximum tree depth
  - Learning rate

## Results

Through experimentation with hyperparameters:
- Increased number of trees (n_estimators) from 10 to 400 significantly improved performance
- Maximum tree depth of 5 provided a good balance between underfitting and overfitting
- Learning rate around 0.3 worked well with 50 estimators

The model identified the following top features by importance:
1. Promo
2. DayOfWeek_1 (Monday)
3. StoreType_b
4. Promo2
5. StoreType_d

## How to Run

1. Clone this repository
2. Install dependencies: `pip install numpy pandas matplotlib seaborn xgboost graphviz lightgbm scikit-learn`
3. Download the data from Kaggle: [Rossmann Store Sales Competition](https://www.kaggle.com/c/rossmann-store-sales)
4. Run the Jupyter notebook to reproduce the analysis
