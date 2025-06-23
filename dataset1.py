import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("c:/Users/sindh/OneDrive/Desktop/Internship/Titanic-Dataset.csv")
print(" Dataset loaded.")


df = df.drop(columns=['Cabin'])
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop_duplicates()
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

print(" Initial cleaning done.")


df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)
print(" Categorical features encoded.")


scaler = StandardScaler()
num_cols = ['age', 'fare', 'sibsp', 'parch']
df[num_cols] = scaler.fit_transform(df[num_cols])
print(" Numerical features standardized.")


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

print(" Showing boxplots:")
for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

print(" Removing outliers...")
for col in num_cols:
    df = remove_outliers(df, col)


df.to_csv("c:/Users/sindh/OneDrive/Desktop/Internship/dataset_final.csv", index=False)
print(" Data cleaned, encoded, scaled, outliers removed, and saved as 'dataset_final.csv'")
