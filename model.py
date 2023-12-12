import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Data preprocessing
df = pd.read_excel('World_development_mesurement.xlsx')
le = LabelEncoder()
le.fit(df['Country'])
df['Country_encoded'] = le.fit_transform(df['Country'])
df['Country_encoded'] = df['Country_encoded'].astype(float)
df.drop(['Country'], axis=1, inplace=True)
columns_to_clean = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound', 'Business Tax Rate']
for column in columns_to_clean:
    df[column] = df[column].str.replace('[$,%]', '', regex=True)
    df[column] = pd.to_numeric(df[column], errors='coerce')
columns_to_fill_mean = ['Business Tax Rate', 'Ease of Business', 'Health Exp % GDP', 'Hours to do Tax', 'Population 0-14','Population Urban']
columns_to_fill_median = ['Birth Rate', 'CO2 Emissions', 'Days to Start Business', 'Energy Usage', 'Health Exp/Capita', 'GDP', 'Infant Mortality Rate', 'Internet Usage', 'Lending Interest', 'Life Expectancy Female', 'Life Expectancy Male', 'Mobile Phone Usage', 'Tourism Inbound', 'Tourism Outbound', 'Population 15-64', 'Population 65+']
for column in columns_to_fill_mean:
    df[column] = df[column].fillna(df[column].mean())
for column in columns_to_fill_median:
    df[column] = df[column].fillna(df[column].median())
df.drop(columns='Number of Records', inplace=True)
df.drop(["Ease of Business"], inplace=True, axis=1)
columns_to_drop = df[["Hours to do Tax", "Business Tax Rate", "Days to Start Business"]]
df.drop(columns_to_drop, inplace=True, axis=1)
df = df.rename(columns={'Birth Rate': 'BirthRate', 'Business Tax Rate': 'BusinessTaxRate','CO2 Emissions':'CO2Emissions','Days to Start Business':'DaystoStartBusiness','Ease of Business':'EaseofBusiness','Energy Usage':'EnergyUsage',
                            'Health Exp % GDP':'HealthExpGDP','Health Exp/Capita':'HealthExpCapita','Hours to do Tax':'HourstodoTax','Infant Mortality Rate':'InfantMortalityRate','Internet Usage':'InternetUsage','Lending Interest':'LendingInterest',
                            'Life Expectancy Female':'LifeExpectancyFemale','Life Expectancy Male':'LifeExpectancyMale','Mobile Phone Usage':'MobilePhoneUsage','Number of Records':'NumberofRecords','Population 0-14':'Population0to14',
                            'Population 15-64':'Population15to64','Population 65+':'Populationmorethan65','Population Total':'PopulationTotal','Population Urban':'PopulationUrban','Tourism Inbound':'TourismInbound','Tourism Outbound':'TourismOutbound'})

# Isolation Forest for outlier detection
iso_forest = IsolationForest(contamination=0.05)
iso_forest.fit(df)
outlier_labels = iso_forest.predict(df)
data_no_outliers = df[outlier_labels != -1]

# Feature scaling and TSNE transformation
scaler = StandardScaler()
df_std = scaler.fit_transform(data_no_outliers)
tsne = TSNE()
data_tsne = tsne.fit_transform(df_std)

# KMeans clustering
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(data_tsne)
y = pd.DataFrame(kmeans.labels_)

# Splitting data for training
X = data_no_outliers
y = y.values.flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network model training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
clf_nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
clf_nn.fit(X_train_scaled, y_train)

# Evaluating model accuracy
y_pred_train = clf_nn.predict(X_train_scaled)
accuracy = accuracy_score(y_train, y_pred_train)
print(f"Model accuracy: {accuracy}")

# Saving the trained model and label encoder
joblib.dump(clf_nn, 'trained_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
