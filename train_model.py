import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# Load data
df = pd.read_csv('netflix_titles.csv')

# Filter for movies
df = df[df['type'] == 'Movie']

# Filter for valid duration
df = df[df['duration'].str.contains(' min', na=False)]

# Extract duration as int
df['duration'] = df['duration'].str.replace(' min', '').astype(int)

# Drop rows with missing values in key columns
df = df.dropna(subset=['release_year', 'rating', 'country', 'duration'])

# Select features
features = ['release_year', 'type', 'rating', 'country']
target = 'duration'

X = df[features]
y = df[target]

# Encode categorical variables
type_encoder = LabelEncoder()
rating_encoder = LabelEncoder()
country_encoder = LabelEncoder()

X['type_encoded'] = type_encoder.fit_transform(X['type'])
X['rating_encoded'] = rating_encoder.fit_transform(X['rating'])
X['country_encoded'] = country_encoder.fit_transform(X['country'])

# Select encoded features
X_encoded = X[['release_year', 'type_encoded', 'rating_encoded', 'country_encoded']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(type_encoder, 'type_encoder.pkl')
joblib.dump(rating_encoder, 'rating_encoder.pkl')
joblib.dump(country_encoder, 'country_encoder.pkl')

print("Model and encoders saved.")
