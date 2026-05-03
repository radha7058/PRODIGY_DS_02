import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

# Basic info
print(df.head())
print(df.info())
print(df.isnull().sum())

# -----------------------------
# DATA CLEANING
# -----------------------------

# Fill missing Age with mean
df["Age"].fillna(df["Age"].mean(), inplace=True)

# Drop Cabin column (too many missing values)
df.drop(columns=["Cabin"], inplace=True)

# Fill Embarked with most frequent value
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# -----------------------------
# EDA (Exploratory Analysis)
# -----------------------------

# 1. Survival count
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.show()

# 2. Survival by Gender
sns.countplot(x="Survived", hue="Sex", data=df)
plt.title("Survival by Gender")
plt.savefig("survival_gender.png")
plt.show()

# 3. Age distribution
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("age_distribution.png")
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.show()