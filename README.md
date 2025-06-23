# 🚢 Titanic Dataset Preprocessing

This project performs data cleaning, encoding, outlier detection/removal, and feature scaling on the Titanic dataset to prepare it for machine learning models.

---

## 📂 Dataset

The dataset used is the famous Titanic dataset, which contains demographic and travel information about the passengers aboard the Titanic.

---

## 🔧 Preprocessing Steps

1. **Data Cleaning**
   - Dropped the `Cabin` column due to too many missing values.
   - Filled missing values:
     - `Age` → with median
     - `Embarked` → with mode
   - Removed duplicate rows.
   - Trimmed whitespace in string columns.
   - Cleaned column names (lowercase, underscores).

2. **Categorical Encoding**
   - One-hot encoded:
     - `Sex` → `sex_male`
     - `Embarked` → `embarked_Q`, `embarked_S`

3. **Outlier Detection & Removal**
   - Visualized numerical columns with boxplots.
   - Removed outliers using IQR method for:
     - `Age`
     - `Fare`
     - `SibSp`
     - `Parch`

4. **Feature Scaling**
   - Applied `StandardScaler` to:
     - `Age`, `Fare`, `SibSp`, `Parch`

5. **Output**
   - Final cleaned dataset saved as:
     ```
     dataset_final.csv
     ```

---

## 📊 Libraries Used

- pandas
- seaborn
- matplotlib
- scikit-learn

---

Data source:
https://www.kaggle.com/datasets/yasserh/titanic-dataset
