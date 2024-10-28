import warnings
warnings.filterwarnings("ignore")
import copy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""# Prepare some functions to use :
##### 1- Function to get the all values of each column
"""
def GetValuesCounts(data):
    import pandas as pd
    columns = [col for col in data.columns ]

    for col in columns:
        print(data[col].value_counts(),"\n")

"""##### 2- Function to get the object (string) Features"""

def GetObjectFeatures(data) :
    object_columns = [col for col in data.columns if data[col].dtype == 'object']

    for col in object_columns :
        print(data[col].value_counts(),"\n")

"""##### 3- Function to "Plot Box Plot" and it works only for the datatype of integer and float columns ."""

def PBP (data):
    for col in data.select_dtypes(include=['int', 'float']).columns:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(f'Boxplot of {col}')
        bp=data.boxplot(column=col)
        plt.show()
        plt.close(fig)

"""##### 4- Function to Plot Pair Plot with different Hues"""

def TribleP(data):
    columns = [col for col in data.columns]
    custom_palette = sns.color_palette("coolwarm", 6)

    for col in columns:
        if data[col].value_counts().count() <= 6:
            sns.pairplot(data, hue=col, palette=custom_palette)
            plt.show()
        else:
            continue

"""##### 5- Function to Plot Distribution Plot"""

def PDP(data):
    columns = [col for col in data.columns]
    colors = sns.color_palette("coolwarm", len(columns))

    for i, col in enumerate(data.select_dtypes(include=['int', 'float']).columns):
        if data[col].value_counts().count() <= 10000:
            sns.histplot(data[col], kde=True, bins=15, color=colors[i % len(colors)])
            plt.title(f'Distribution of {col}')
            plt.show()

"""##### 6- Function to Plot Pie chart if the feature has values less than or equal 10"""

def PlotPie(data):
    custom_palette = "coolwarm"
    sns.set_palette(custom_palette)


    plt.figure(figsize=(8, 6))

    columns = [col for col in data.columns]

    for col in columns:
        if data[col].value_counts().count() <= 10:
            # Plot pie chart
            plt.figure(figsize=(8,6))
            data[col].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140)
            plt.title(f"Distribution of {col}")
            plt.ylabel("")
            plt.axis('equal')
            plt.show()
            print("\n\n\n")

data = pd.read_excel("heart_data.xlsx")
copiedData = copy.deepcopy(data)

print(data.shape)

# Filter the rows where 'HadHeartAttack' is 0
zero_rows = data[data['HadHeartAttack'] == 0]

# Drop the first 200,000 rows with 0 in 'HadHeartAttack'
rows_to_drop = zero_rows.index[:200000]
data = data.drop(rows_to_drop)

# Check the new shape of the data to confirm
print(f"New shape of the dataset: {data.shape}")

"""# EDA & Statistical Analysis :"""

data.head(7)

data.tail(7)

data.info()

data.columns

data.describe()

data.shape

NumiricalColumns=['HeightInMeters', 'WeightInKilograms', 'BMI','HadHeartAttack',
       'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
       'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis','DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
       'DifficultyConcentrating', 'DifficultyWalking',
       'DifficultyDressingBathing', 'DifficultyErrands','ChestScan','HighRiskLastYear', 'CovidPos','AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver']
for col in NumiricalColumns :
    print("Variance for",col,"column :",data[col].var())

for col1 in NumiricalColumns :
    for col2 in NumiricalColumns :
        if (col1==col2):
            continue
        else:
            print("Correlation for",col1,"column with",col2,"column :",data[col].corr(data[col2]))

for col in NumiricalColumns:
    print("Skewness for", col, "column:", data[col].skew())

data.isna().sum()

data.duplicated().sum()

GetValuesCounts(data)

GetObjectFeatures(data)

PBP(data)

PDP(data)

PlotPie(data)

plt.figure(figsize=(10, 6))
sns.countplot(x='HadHeartAttack', data=data)
plt.title('Count of Yes  vs No')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

count_of_ones = data['HadHeartAttack'].value_counts()[1]
print(f"Number of 1s in HadHeartAttack column: {count_of_ones}")

count_of_ones = data['HadHeartAttack'].value_counts()[0]
print(f"Number of 0s in HadHeartAttack column: {count_of_ones}")

plt.figure(figsize=(10, 8))
correlation_matrix = data[['HeightInMeters', 'WeightInKilograms', 'BMI','HadHeartAttack',
       'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
       'DifficultyConcentrating']].astype(float).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = data[[
       'DifficultyConcentrating', 'DifficultyWalking',
       'DifficultyDressingBathing', 'DifficultyErrands','ChestScan','HighRiskLastYear', 'CovidPos','AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver']].astype(float).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# Define columns
numerical_cols = [ 'HeightInMeters', 'WeightInKilograms','BMI']
categorical_cols = [
    'Sex', 'GeneralHealth', 'AgeCategory', 'SmokerStatus',
    'ECigaretteUsage', 'HadDiabetes','AlcoholDrinkers'
]

def load_data():
    # Load the data
    data = pd.read_excel("heart_data.xlsx")
    zero_rows = data[data['HadHeartAttack'] == 0]
    rows_to_drop = zero_rows.index[:200000]
    data = data.drop(rows_to_drop)

    cols_to_drop = [
        'PatientID', 'State', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
        'HadArthritis','HadKidneyDisease', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
        'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing',
        'DifficultyErrands','RaceEthnicityCategory','ChestScan', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
        'TetanusLast10Tdap', 'HighRiskLastYear'
    ]
    df_cleaned = data.drop(columns=cols_to_drop)

    global label_encoder, scaler
    label_encoder = {}

    # Encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col])
        label_encoder[col] = le  # Store the label encoder for future use

    # Initialize and fit the scaler
    scaler = StandardScaler()
    df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])

    X = df_cleaned.drop('HadHeartAttack', axis=1)
    y = df_cleaned['HadHeartAttack']

    return X, y


def train_models(X, y):
    global trained_models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    trained_models = {}

    # Train SVM
    svm_model = SVC(kernel='rbf', gamma=0.01, C=10)
    svm_model.fit(X_train_resampled, y_train_resampled)
    trained_models['SVM'] = svm_model

    # Train Logistic Regression
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)
    trained_models['Logistic Regression'] = logistic_model

    # Train Random Forest
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train_resampled, y_train_resampled)
    trained_models['Random Forest'] = random_forest_model

    # Train XGBoost
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_resampled, y_train_resampled)
    trained_models['XGBoost'] = xgb_model

    # Calculate accuracies
    model_accuracies = {}
    for model_name, model in trained_models.items():
        if model_name == 'Logistic Regression':
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[model_name] = accuracy

    return trained_models, model_accuracies


def predict_models(input_data, trained_models, model_accuracies):
    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in categorical_cols:
        if col in label_encoder:
            input_df[col] = label_encoder[col].transform(input_df[col])

    # Scale numerical columns
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    model_prediction = []
    for model_name, model in trained_models.items():
        pred = model.predict(input_df)
        model_prediction.append(pred[0])

    yes_models = [model for i, model in enumerate(trained_models.keys()) if model_prediction[i] == 1 ]
    no_models = [model for i, model in enumerate(trained_models.keys()) if model_prediction[i] == 0]
    total_yes_accuracy = sum(model_accuracies[model] for model in yes_models)

    total_no_accuracy = sum(model_accuracies[model] for model in no_models)

    if len(yes_models) ==0:
        avg_yes_accuracy = 0

    else:
        avg_yes_accuracy = total_yes_accuracy / len(yes_models)

    if len(no_models) == 0:
        avg_no_accuracy = 0

    else:
        avg_no_accuracy = total_no_accuracy / len(no_models)

    if avg_yes_accuracy > avg_no_accuracy:
        prediction = "Heart Attack Risk: Yes "
    else:
        prediction = "Heart Attack Risk: No "

    return prediction
'''
# Load data and train models
X, y = load_data()
trained_models, model_accuracies = train_models(X, y)

# Example test input
test_input = {
    'Sex': 'Male',
    'GeneralHealth': 'Good',
    'AgeCategory': 'Age 45 to 49',
    'HeightInMeters': 1.75,
    'WeightInKilograms': 70,
    'BMI': 25.0,
    'HadAngina': 0,  # Add value as needed
    'HadStroke': 0,
    'HadDepressiveDisorder': 0,  # Add value as needed
    'HadDiabetes': 'No',
    'SmokerStatus': 'Never smoked',
    'ECigaretteUsage': 'Never used e-cigarettes in my entire life',
    'AlcoholDrinkers': '0',
    'CovidPos': 0  # Add value as needed
}

# Call the predict_models function
predictions=predict_models(test_input, trained_models, model_accuracies)
print("Test Predictions:", predictions)
'''