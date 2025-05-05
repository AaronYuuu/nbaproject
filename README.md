# Chris ~~Paul~~ Parlay
As I learned about data science and machine learning I wanted a project that would let me practice and challenge my skills. As an avid drop-in basketball player and sports bettor, I wanted to do something NBA related. For the project I found statistics on home and away teams for _almost_ everygame in the 2023-2024 season: (https://www.kaggle.com/datasets/wyattowalsh/basketball/data).<br/>
Completing this project helped me learn a lot about the data pipeline and the processes that can be used in feature engineering and filtering data. I decided on using a random forest classifier over other models because it limits overfitting due to having less tress, allowing for more accurate predictions and was also begineer friendly. <br/>
<br/>
### Sample output predicting a matchup between the LA Lakers and Boston Celtics: 
![image](https://github.com/user-attachments/assets/e9626864-bbb9-49ba-aa22-1ba22ca97acc)

## About the branches:
- **nba.py**: my final model
- **-.csv**: datasets used after filtering with excel
- **gamePrediction.ipynb**: Jupyter notebook I used to test out code
  

# How It’s Made

**Tech used:** Python, Jupyter Notebook  
**Jupyter notebook:**
- Used for trouble-shooting and debugging
**Python:**
- Everything else
  
## Data ingestion & cleaning 
  - Used **pandas** to load `teamPerGame.csv` (season averages) and `2024-games.csv` (historical game results).  
  - Wrote `get_stats()` to map full team names (with stray “*” characters) to NBA abbreviations via a dictionary lookup.  

## Feature engineering 
  - Implemented `compute_differential_features()` to calculate the difference in key metrics (FGA, FG%, 3PA, 3P%, FTA, FT%, ORB, DRB, STL, BLK, TOV) between home and away teams.  
  - Centralized column renaming in `rename_columns()` to keep feature names consistent (e.g. `FG%_diff` → `fg_pct_diff`).  

## Model training & evaluation
  - Built `train_model()` which:  
    1. Reads historical game data, labels home wins vs. away wins.  
    2. Constructs a design matrix of differential features for each game.  
    3. Splits into train/test sets with `sklearn.model_selection.train_test_split`.  
    4. Trains a `RandomForestClassifier(n_estimators=100, random_state=42)`.  
    5. Reports test-set accuracy using `sklearn.metrics`.  

## Prediction script  
  - The `main()` function ties it all together:  
    - Loads current‐season stats.  
    - Computes the feature differential for a specific matchup (e.g. LAL vs. BOS).  
    - Trains the model.  
    - Prints out the home‐team win probability via `predict_win_probability()`.
# Lessons Learned
1. Feature engineering matters: isolating the right set of statistics and computing head-to-head differentials proved far more effective than raw per-team averages, finding an **effective** dataset made this a lot easier and streamlined my collection process. Next time I would spend more time in the preliminary phase. Learning how to scrape data myself would have helped immensely. 
2. Data hygiene is critical: handling quirks in team naming (trailing “*” characters) and missing values taught me to build robust preprocessing steps first and to reduce the confusion of variable names.
3. Model evaluation workflow: splitting data, training, predicting, and computing accuracy reinforced best practices in building ML pipelines end to end.
4. Performance tuning: seeing how small refactors in pandas can yield big speedups inspired me to always profile data-processing code before scaling up
5. Spending more time brainstorming before diving into a project can be very helpful and would have saved me a lot of time scrambling around looking for a dataset

# What I'm Still Working On
- Implemting seaborn and matplotlib to graph out heatmaps and highlight the correlation between specific varaibles allowing me to fine-tune features
- Expanding to other machine learning models that may be more applicable to predicting sports games as there are many hidden variables
- Expanding and combining mutiple datasets to improve features
- Improving the organization and readability of my code
