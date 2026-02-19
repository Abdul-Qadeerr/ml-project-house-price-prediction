# House Price Prediction in Pakistan  

## Project Overview
This project aims to predict house prices in Pakistan using various machine learning algorithms. The dataset is sourced from Zameen.com, a leading real estate platform in Pakistan, containing property listings from major cities including Islamabad, Karachi, Lahore, Faisalabad, and Rawalpindi.

## Dataset Information
- **Source**: Zameen.com (Entities.csv)
- **Total Records**: 168,446 property listings
- **Features**: Property type, location, price, number of bedrooms/bathrooms, area, purpose, etc.

## Project Structure

### 1. Problem Statement
Prediction of house prices in Pakistan using various property attributes and machine learning models.

### 2. Libraries Used
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sklearn.linear_model as linMod
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
```

### 3. Data Collection
- Data imported from 'Entities.csv' file
- Initial dataset contains 168,446 entries with 18 columns

### 4. Data Wrangling
**Data Cleaning Steps:**
- Removed irrelevant columns: 'Unnamed: 0', 'property_id', 'location_id', 'page_url', 'location', 'date_added', 'agency', 'agent'
- Handled missing values
- Renamed columns for better readability
- Removed outliers using z-score method (threshold = 3)

**Outlier Removal:**
```python
housesClean = housesClean[(np.abs(stats.zscore(housesClean['area'])) < 3)]
housesClean = housesClean[(np.abs(stats.zscore(housesClean['bedrooms'])) < 3)]
housesClean = housesClean[(np.abs(stats.zscore(housesClean['baths'])) < 3)]
housesClean = housesClean[(np.abs(stats.zscore(housesClean['price'])) < 3)]
```
After outlier removal: **164,214 records** remaining

### 5. Data Visualization
The project includes various visualizations:
- Pair plots showing relationships between features
- Scatter plots for:
  - Bedrooms vs Price
  - Bathrooms vs Price  
  - Area vs Price (with regression line)
  - City vs Price
  - Province vs Price
  - Property Type vs Price

- **Heatmap** showing correlation between all features

### 6. Feature Engineering
Categorical variables were converted to numerical using one-hot encoding:
- property_type → 6 binary columns
- purpose → binary column
- city → multiple binary columns  
- province → multiple binary columns

**Final features count: 19 columns**

### 7. Machine Learning Models

#### Data Split
- Training set: 80% of data
- Testing set: 20% of data
- Features (X): All columns except 'price'
- Target (y): 'price' column

#### Model 1: Multiple Linear Regression
- **R² Score: 0.365**
- Baseline model for comparison

#### Model 2: Decision Tree Regression
- **R² Score: 0.891**
- Strong performance, capturing non-linear relationships

#### Model 3: Random Forest Regression
- **R² Score: 0.925**
- Best performing model
- Ensemble method reduces overfitting

### 8. Model Evaluation

| Model | R² Score | Performance |
|-------|----------|-------------|
| Multiple Linear Regression | 0.365 | Poor |
| Decision Tree | 0.891 | Good |
| Random Forest | 0.925 | Excellent |

## Key Insights
1. **Random Forest** performed best with 92.5% accuracy
2. **Area** and **number of bedrooms/bathrooms** are strong predictors of price
3. Property location (city/province) significantly impacts pricing
4. Outlier removal improved model performance

## Usage

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Running the Project
1. Ensure 'Entities.csv' is in the working directory
2. Run all cells in the Jupyter notebook sequentially
3. Models will train automatically and display results

## Future Improvements
- Feature engineering (creating area per room ratios)
- Hyperparameter tuning for Random Forest
- Try gradient boosting algorithms (XGBoost, LightGBM)
- Include more geographical features
- Time-based analysis (date_added column)

## Files in Repository
- `Entities.csv` - Raw dataset
- Jupyter notebook with complete analysis
- This README file

## Acknowledgments
- Data source: Zameen.com
- Project completed as part of machine learning coursework
