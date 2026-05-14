import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# ---------------- LOAD DATA ----------------
df = pd.read_csv("crime_dataset_india (1).csv")

# ---------------- VISUALIZATION ----------------
df['City'].value_counts().head(10).plot(kind='bar')
plt.title("Top Crime Cities")
plt.xlabel("City")
plt.ylabel("Crime Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------- PREPROCESSING ----------------
df['Date_of_occurrence'] = pd.to_datetime(df['Date_of_occurrence'], errors='coerce')

df['Hour'] = pd.to_datetime(
    df['Time_of_occurrence'],
    format='%H:%M',
    errors='coerce'
).dt.hour

df['Year'] = df['Date_of_occurrence'].dt.year
df['Month'] = df['Date_of_occurrence'].dt.month
df['Day'] = df['Date_of_occurrence'].dt.day
df['Weekday'] = df['Date_of_occurrence'].dt.weekday

# Age grouping (advanced feature)
df['Age_Group'] = pd.cut(
    df['Victim_Age'],
    bins=[0,18,30,50,100],
    labels=[0,1,2,3]
)

# Select useful columns
df = df[['City','Crime_Description','Victim_Age','Victim_Gender',
         'Weapon_Used','Crime_Domain','Year','Month','Hour','Age_Group']]

df = df.dropna()

# ---------------- ENCODING ----------------
le = {}
for col in ['City','Crime_Description','Victim_Gender','Weapon_Used','Crime_Domain']:
    le[col] = LabelEncoder()
    df[col] = le[col].fit_transform(df[col])

# ---------------- CLUSTERING ----------------
cluster_data = df[['City', 'Victim_Age']]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# ---------------- FEATURES ----------------
X = df.drop('Crime_Description', axis=1)
y = df['Crime_Description']

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# ---------------- CLUSTER VISUAL ----------------
print("\nCluster Distribution:")
print(df['Cluster'].value_counts())

# ---------------- PREDICTION ----------------
print("\n--- Predict Crime ---")

city = input("City: ")
age = int(input("Age: "))
gender = input("Gender (M/F/X): ")
weapon = input("Weapon: ")
domain = input("Crime Domain: ")
year = int(input("Year: "))
month = int(input("Month: "))
hour = int(input("Hour (0-23): "))

try:
    city_enc = le['City'].transform([city])[0]
    gender_enc = le['Victim_Gender'].transform([gender])[0]
    weapon_enc = le['Weapon_Used'].transform([weapon])[0]
    domain_enc = le['Crime_Domain'].transform([domain])[0]

    age_group = 0 if age <= 18 else 1 if age <= 30 else 2 if age <= 50 else 3

    input_data = [[
        city_enc,
        age,
        gender_enc,
        weapon_enc,
        domain_enc,
        year,
        month,
        hour,
        age_group,
        0   # placeholder cluster
    ]]

    pred = model.predict(input_data)[0]
    result = le['Crime_Description'].inverse_transform([pred])[0]

    print("\nPredicted Crime:", result)

except:
    print("Invalid input (check spelling)")