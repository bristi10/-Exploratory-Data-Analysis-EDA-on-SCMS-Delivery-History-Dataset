import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns',None)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"]=(12,6)

df=pd.read_csv("SCMS_Delivery_History_Dataset.csv")

print(df.head())

print(df.shape)


print(df.info())

missing=df.isnull().sum().sort_values(ascending=False)
print(missing)

plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing values")
plt.show()

duplicate= df.duplicated().sum()
print(duplicate)

num_cols=df.select_dtypes(include=['int64','float64']).columns
print(num_cols)

cat_cols=df.select_dtypes(include=['object']).columns
print(cat_cols)

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col],kde=True)
    plt.title(col)
    plt.show()

for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(
            y=df[col],
            data=df,
            order=df[col].value_counts().index[:10]
        )
    plt.title(col)
    plt.show()

sns.scatterplot(
   x="Weight (Kilograms)",
   y="Line Item Value",
   data=df
)
plt.title("Bivariate Analysis")
plt.show()

date_cols=[
    'Scheduled Delivery Date',
    'Delivered to Client Date',
    'Delivery Recorded Date'
]

for col in date_cols:
    df[col]=pd.to_datetime(df[col], errors='coerce')

df['Delivery Month']=df['Scheduled Delivery Date'].dt.month_name()



plt.figure(figsize=(6,4))
sns.countplot(
    y='Delivery Month',
    data=df,
    order=df['Delivery Month'].value_counts().index
)  
plt.show()  

top_country=df['Country'].value_counts().head(10)

plt.figure(figsize=(8,5))
top_country.plot(kind='bar')
plt.title("Top 10 Countries")
plt.xlabel("Country")
plt.ylabel("Count")
plt.show()

top_vendor=df['Vendor'].value_counts().head(10)

plt.figure(figsize=(8,5))
top_vendor.plot(kind='bar')
plt.title("Top 10 Vendors")
plt.xlabel("Vendor")
plt.ylabel("Count")
plt.show()

   