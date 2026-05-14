import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="CrimeScope", layout="wide")
st.title("🚔 CrimeScope – Intelligent Crime Prediction & Visualization")

# ---------------- LOAD DATA (CACHED) ----------------
@st.cache_data
def load_data():
    return pd.read_csv("crime_dataset_india (1).csv")

df = load_data()

# ---------------- PREPROCESS + TRAIN (CACHED) ----------------
@st.cache_resource
def process_and_train(df):

    df['Date_of_occurrence'] = pd.to_datetime(df['Date_of_occurrence'], errors='coerce')
    df['Hour'] = pd.to_datetime(df['Time_of_occurrence'], format='%H:%M', errors='coerce').dt.hour
    df['Year'] = df['Date_of_occurrence'].dt.year
    df['Month'] = df['Date_of_occurrence'].dt.month

    df['Age_Group'] = pd.cut(df['Victim_Age'],
                            bins=[0,18,30,50,100],
                            labels=[0,1,2,3])

    df = df[['City','Crime_Description','Victim_Age','Victim_Gender',
             'Weapon_Used','Crime_Domain','Year','Month','Hour','Age_Group']]

    df = df.dropna()

    # Encoding
    le = {}
    for col in ['City','Crime_Description','Victim_Gender','Weapon_Used','Crime_Domain']:
        le[col] = LabelEncoder()
        df[col] = le[col].fit_transform(df[col])

    # Clustering
    cluster_data = df[['City','Victim_Age']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_data)

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled)

    # Model
    X = df.drop('Crime_Description', axis=1)
    y = df['Crime_Description']

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y)

    return df, model, le, X

df, model, le, X = process_and_train(df)

# ---------------- ACCURACY ----------------
y_pred = model.predict(X)
acc = accuracy_score(df['Crime_Description'], y_pred)
st.write(f"📊 Model Accuracy: {round(acc*100,2)}%")

# ---------------- UI ----------------
st.header("🔮 Predict Crime")

city = st.selectbox("City", le['City'].classes_)
age = st.slider("Age", 10, 80, 25)
gender = st.selectbox("Gender", le['Victim_Gender'].classes_)
weapon = st.selectbox("Weapon Used", le['Weapon_Used'].classes_)
domain = st.selectbox("Crime Domain", le['Crime_Domain'].classes_)
year = st.number_input("Year", value=2020)
month = st.slider("Month", 1, 12, 1)
hour = st.slider("Hour", 0, 23, 12)

if st.button("Predict Crime"):
    age_group = 0 if age <= 18 else 1 if age <= 30 else 2 if age <= 50 else 3

    input_df = pd.DataFrame([[
        le['City'].transform([city])[0],
        age,
        le['Victim_Gender'].transform([gender])[0],
        le['Weapon_Used'].transform([weapon])[0],
        le['Crime_Domain'].transform([domain])[0],
        year,
        month,
        hour,
        age_group,
        0
    ]], columns=X.columns)

    pred = model.predict(input_df)[0]
    result = le['Crime_Description'].inverse_transform([pred])[0]

    st.success(f"🚨 Predicted Crime: {result}")

# ---------------- INSIGHTS ----------------
st.header("📊 Insights Dashboard")

st.subheader("Top Crime Cities")
st.bar_chart(df['City'].value_counts().head(10))

st.subheader("Cluster Distribution")
st.bar_chart(df['Cluster'].value_counts())

# ---------------- MAP ----------------
st.header("🗺 Crime Hotspots Map")

city_coords = {
    "Delhi": [28.61, 77.23],
    "Mumbai": [19.07, 72.87],
    "Bangalore": [12.97, 77.59],
    "Hyderabad": [17.38, 78.48],
    "Chennai": [13.08, 80.27],
    "Pune": [18.52, 73.85],
    "Kolkata": [22.57, 88.36]
}

m = folium.Map(location=[20.59, 78.96], zoom_start=5)

for city_val in df['City'].unique():
    original_city = le['City'].inverse_transform([city_val])[0]
    if original_city in city_coords:
        folium.CircleMarker(
            location=city_coords[original_city],
            radius=5,
            popup=original_city,
            color="red",
            fill=True
        ).add_to(m)

st_folium(m, width=700, height=500)