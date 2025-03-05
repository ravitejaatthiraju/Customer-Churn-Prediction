import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your data (replace 'customer_data.csv' with your file)
try:
    df = pd.read_csv('customer_data.csv')
except FileNotFoundError:
    print("Error: customer_data.csv not found. Please provide your data.")
    exit()

# Debugging: Inspect the DataFrame
print("Columns in the DataFrame:", df.columns)
print("First 5 rows of the DataFrame:\n", df.head())

# 2. Data Exploration and Preprocessing
# (Add more exploration and cleaning based on your data)

# Example: Handle missing values (imputation)
numerical_features = df.select_dtypes(include=['number']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
# Ensure 'churn' is not included in numerical or categorical features
if 'churn' in numerical_features:
    numerical_features.remove('churn')
if 'churn' in categorical_features:
    categorical_features.remove('churn')

numerical_transformer = Pipeline(steps=[
   ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), #impute before one hot encoding
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# Separate features (X) and target (y)
X = df.drop('churn', axis=1)  # Assuming 'churn' is your target column
y = df['churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 3. Model Training and Evaluation

# Logistic Regression
logistic_model = LogisticRegression(solver='liblinear', random_state=42)
logistic_model.fit(X_train_processed, y_train)
logistic_predictions = logistic_model.predict(X_test_processed)

print("Logistic Regression:")
print(classification_report(y_test, logistic_predictions))
print(confusion_matrix(y_test, logistic_predictions))

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_processed, y_train)
rf_predictions = rf_model.predict(X_test_processed)

print("\nRandom Forest:")
print(classification_report(y_test, rf_predictions))
print(confusion_matrix(y_test, rf_predictions))

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_processed, y_train)
gb_predictions = gb_model.predict(X_test_processed)

print("\nGradient Boosting:")
print(classification_report(y_test, gb_predictions))
print(confusion_matrix(y_test, gb_predictions))

# 4. Feature Importance (for Random Forest and Gradient Boosting)

if isinstance(rf_model, RandomForestClassifier) and hasattr(rf_model, 'feature_importances_'):
    feature_importance_rf = rf_model.feature_importances_
    feature_names_rf = preprocessor.get_feature_names_out()
    feature_importance_df_rf = pd.DataFrame({'Feature': feature_names_rf, 'Importance': feature_importance_rf})
    feature_importance_df_rf = feature_importance_df_rf.sort_values(by='Importance', ascending=False)
    print("\nRandom Forest Feature Importance:")
    print(feature_importance_df_rf.head(10)) #display top 10

if isinstance(gb_model, GradientBoostingClassifier) and hasattr(gb_model, 'feature_importances_'):
    feature_importance_gb = gb_model.feature_importances_
    feature_names_gb = preprocessor.get_feature_names_out()
    feature_importance_df_gb = pd.DataFrame({'Feature': feature_names_gb, 'Importance': feature_importance_gb})
    feature_importance_df_gb = feature_importance_df_gb.sort_values(by='Importance', ascending=False)
    print("\nGradient Boosting Feature Importance:")
    print(feature_importance_df_gb.head(10)) #display top 10

# 5. Visualization (example)
# (Create more visualizations based on your data and analysis)
sns.countplot(x='churn', data=df)
plt.title('churn Distribution')
plt.show()

numerical_df = df[numerical_features + ['churn']]  # add churn column to numerical only dataframe

# Convert 'churn' to numeric (1/0) if needed
if numerical_df['churn'].dtype == 'object':
    numerical_df['churn'] = numerical_df['churn'].map({'Yes': 1, 'No': 0})

print(numerical_df.dtypes)
print(numerical_df.head())

numerical_df = numerical_df.apply(pd.to_numeric, errors='coerce')
numerical_df.fillna(numerical_df.mean(), inplace=True)

correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()