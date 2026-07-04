import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_excel('INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8 (1).xls')

# Prepare features and target
X = df[[
    'EmpEnvironmentSatisfaction',
    'EmpLastSalaryHikePercent',
    'ExperienceYearsInCurrentRole',
    'ExperienceYearsAtThisCompany',
    'YearsSinceLastPromotion'
]]

y = df['PerformanceRating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Save model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("Model trained and saved successfully!")
print(f"Training accuracy: {rf_model.score(X_train, y_train):.4f}")
print(f"Testing accuracy: {rf_model.score(X_test, y_test):.4f}")
