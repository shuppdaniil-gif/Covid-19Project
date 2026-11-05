import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import clone
from imblearn.over_sampling import SMOTE

df = pd.read_csv('covid_data.csv')

df['date'] = pd.to_datetime(df['date'])

print(df.head(5))
print(df.info())

duplicate = df.duplicated().sum()
print(duplicate)
df = df.drop_duplicates()

misses = df.isnull().sum()
print(misses)

df_continent = df[df['continent'].notna()].copy()

df_continent['total_cases'] = df_continent['total_cases'].fillna(0)
df_continent['new_cases'] = df_continent['new_cases'].fillna(0)
df_continent['total_deaths'] = df_continent['total_deaths'].fillna(0)
df_continent['new_deaths'] = df_continent['new_deaths'].fillna(0)
df_continent['new_cases_smoothed'] = df_continent['new_cases_smoothed'].fillna(0)
df_continent['new_deaths_smoothed'] = df_continent['new_deaths_smoothed'].fillna(0)

df_continent.to_csv('covid_clean.csv', index=False)

print('total_cases')
print(df_continent['total_cases'].describe())
print('total_deaths')
print(df_continent['total_deaths'].describe())
#метод дескрайб вказує кількість записів середнє значення стардартне відхилення
#мінімальне та максимальне значення і квантилі розподілу

corr_columns = [
    'total_cases',
    'new_cases',
    'total_deaths',
    'new_deaths',
    'total_vaccinations',
    'people_vaccinated',
    'population',
    'population_density',
    'median_age',
    'gdp_per_capita',
    'life_expectancy',
    'human_development_index'
]

corr_matrix = df_continent[corr_columns].corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('COVID-19 Correlation Matrix')
plt.show()

corr_pairs = corr_matrix.unstack()
corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]

corr_pairs = corr_pairs.drop_duplicates()

print('Strongest correlations:')
print(corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index).head(3))

print('Weakest correlations:')
print(corr_pairs.loc[corr_pairs.abs().sort_values().head(3).index])

df_Ukraine = df_continent[df_continent['location'] == 'Ukraine']

sns.lineplot(
    data=df_Ukraine,
    x='date',
    y='total_cases'
)
plt.show()

latest_date = df_continent[df_continent['total_cases'] > 0]['date'].max()

df_latest = df_continent[df_continent['date'] == latest_date]

df_top10 = df_latest.sort_values(by='total_cases', ascending=False).head(10)

plt.figure(figsize= (10, 6))

sns.barplot(
    data=df_top10,
    x='total_cases',
    y='location',
    palette='viridis',
)

plt.show()

countries = ['Ukraine', 'Japan', 'United States', 'Brazil', 'South Africa']

for country in countries:

    country_data = df_continent[
        df_continent['location'] == country
    ].sort_values('date')

    plt.figure(figsize=(12, 5))

    sns.lineplot(
        data=country_data,
        x='date',
        y='new_cases_smoothed'
    )

    plt.title(f'New COVID Cases - {country}')
    plt.show()

    plt.figure(figsize=(12, 5))

    sns.lineplot(
        data=country_data,
        x='date',
        y='new_deaths_smoothed'
    )

    plt.title(f'New COVID Deaths - {country}')
    plt.show()


vaccination_data = df_continent[
    df_continent['location'].isin(countries)
].copy()

vaccination_data = vaccination_data.dropna(
        subset=['total_vaccinations']
)

fig = px.line(
    vaccination_data,
    x='date',
    y='total_vaccinations',
    color='location',
    title='COVID-19 Vaccination Dynamics',
    labels={
        'date': 'Date',
        'total_vaccinations': 'Total Vaccinations',
        'location': 'Country'
    }
)

fig.show()

df_ml = pd.read_csv('covid_clean.csv')

df_ml['high_cases'] = (df_ml['new_cases'] > 1000).astype(int)

print(df_ml.head())
print(df_ml['high_cases'].value_counts())

y = df_ml['high_cases']

columns_to_drop = ['high_cases', 'new_cases', 'date', 'location']
X = df_ml.drop(columns=columns_to_drop)

print(X.shape)
print(y.shape)
print(X.head())
print(y.head())


continent_encoded = pd.get_dummies(X['continent'])

print(continent_encoded.head())
print(continent_encoded.shape)


X = X.drop(columns='continent')
X = pd.concat((X, continent_encoded), axis=1)

print(X.shape)
print(X.head())


encoder = LabelEncoder()

X['iso_code'] = encoder.fit_transform(X['iso_code'])

print(X['iso_code'].head())
print(X['iso_code'].dtype)

X = X.drop(columns='tests_units')
print(X.shape)
print(X.dtypes.value_counts())

columns_to_drop = X.columns[
    (X.isnull().mean() > 0.8) &
    (X.columns != 'total_vaccinations')
]

print(columns_to_drop)
print(len(columns_to_drop))

X = X.drop(columns=columns_to_drop)
print(X.shape)
print(X.isnull().sum().sum())

missing = X.isnull().sum().sort_values(ascending=False)

print(missing[missing > 0])

bool_columns = X.select_dtypes(include='bool').columns
X[bool_columns] = X[bool_columns].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

imputer = SimpleImputer(strategy='median')

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

scaler = StandardScaler()

scale_columns = [
    'total_cases',
    'total_deaths',
    'total_vaccinations'
]

X_train[scale_columns] = scaler.fit_transform(X_train[scale_columns])
X_test[scale_columns] = scaler.transform(X_test[scale_columns])

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = {}

for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    predictions = model.predict(X_test)

    results[name] = {
        'model': model,
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'f1': f1_score(y_test, predictions, zero_division=0)
    }

    print('\nModel:', name)
    print('Accuracy:', results[name]['accuracy'])
    print('Precision:', results[name]['precision'])
    print('Recall:', results[name]['recall'])
    print('F1:', results[name]['f1'])
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=0))

results_df = pd.DataFrame({
    name: {
        'Accuracy': data['accuracy'],
        'Precision': data['precision'],
        'Recall': data['recall'],
        'F1': data['f1']
    }
    for name, data in results.items()
}).T

print('\nModel comparison:')
print(results_df.sort_values(by='F1', ascending=False))

best_model_name = max(results, key=lambda name: results[name]['f1'])
best_model = results[best_model_name]['model']

print('\nBest model:', best_model_name)

selector = SelectKBest(score_func=f_classif, k=10)

selector.fit(X_train, y_train)

X_train_selected = selector.transform(X_train_smote)
X_test_selected = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]

print('\nSelected features:')
print(selected_features)

selected_model = clone(best_model)
selected_model.fit(X_train_selected, y_train_smote)

selected_predictions = selected_model.predict(X_test_selected)

print('\nResults after Feature Selection:')
print('Accuracy:', accuracy_score(y_test, selected_predictions))
print('Precision:', precision_score(y_test, selected_predictions, zero_division=0))
print('Recall:', recall_score(y_test, selected_predictions, zero_division=0))
print('F1:', f1_score(y_test, selected_predictions, zero_division=0))
print(confusion_matrix(y_test, selected_predictions))
print(classification_report(y_test, selected_predictions, zero_division=0))