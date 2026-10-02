# House Price Prediction

Regression models that predict house sale prices from property features.

**Dataset:** "House Price Prediction - Cleaned Real Estate Dataset with Engineered Pricing Metrics" by Debayan Bandyopadhyay (Kaggle, MIT license). 4,600 Washington State property sales.

## What I did
- Removed records with a price of 0 and extreme price outliers (outside the 1st-99th percentile): 4,600 to 4,460 rows
- Dropped `price_per_sqft`, because it is calculated from the price and would leak the answer
- Built a Scikit-learn pipeline: median/most-frequent imputing, scaling, one-hot encoding of city and zip code
- 70/30 train-test split

## Results (test set)
| Model | R2 | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.775 | 87,014 | 130,393 |
| Random Forest | 0.689 | 96,500 | 153,359 |
| KNN (k=5) | 0.63 | 108,502 | 167,262 |

Linear Regression performed best here. Random Forest used default settings and was not tuned.

## Tools
Python, Pandas, NumPy, Scikit-learn, Google Colab
