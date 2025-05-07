import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Load the cleaned dataset
df = pd.read_csv("clean_data/cleaned_players.csv")

# print("📊 Sample data:")
# print(df.head())

# Define features and target
X = df[['position', 'team', 'Year']]     # Features (categorical)
y = df['adp']                    # Target

# One-hot encode categorical features
categorical_features = ['position', 'team']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Define model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate model performance 
score = pipeline.score(X_test, y_test)
print(f"Model R² score on test set: {score:.2f}")

os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, 'models/draft_model.pkl')
print("✅ Model saved to models/draft_model.pkl")