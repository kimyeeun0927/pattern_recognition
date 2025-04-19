
# News Popularity Prediction

This project was developed as part of the **Pattern Recognition** course at Ewha Womans University.  
We build a binary classification model to predict whether an online news article will become popular based on its features.

##  Objective
Predict if a news article will receive **more than 1,400 shares**.  
The target variable `y` is binary:
- `1` = Popular  
- `0` = Not popular

##  Dataset
- `train.csv`: 22,200 samples, 46 input features  
- `test.csv`: 9,515 samples, 46 input features  
- About 10% missing values per feature

##  Workflow
1. **Exploratory Data Analysis (EDA)**  
2. **Data Preprocessing**: Missing values, normalization, encoding  
3. **Model Training**: Tried multiple machine learning algorithms  
4. **Evaluation**: Accuracy, F1 score, AUC  
5. **Submission Files**:  
   - `code.ipynb`  
   - `prediction.csv` (includes `y_predict`, `y_prob`)  
   - `report.pdf`


##  Team Members
- 김도윤, 김예은, 우민하, 이지원
