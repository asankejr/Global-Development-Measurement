import streamlit as st
import joblib
import pandas as pd

# Selected columns for user input
user_input_columns = [
    "GDP", "CO2Emissions", "HealthExpCapita", "InternetUsage",
    "LifeExpectancyFemale", "LifeExpectancyMale", "Population15to64",
    "TourismInbound", "TourismOutbound", "Country_encoded"
]

def user_input_features():
    st.sidebar.header('User Input Features')
    data = {}
    for column in user_input_columns[:-1]:  # Excluding the last column (Country_encoded)
        value = st.sidebar.number_input(f"Insert the {column}", step=0.1)
        data[column] = value
    country = st.sidebar.text_input("Insert the Country")
    return data, country

def main():
    st.title('Global Development Measurement')
    data, country = user_input_features()
    submitted = st.sidebar.button('Submit')  # Submit button
    if submitted:
        st.header("User input parameters")
        for column, value in data.items():
            st.write(f"{column} = {value}")
        st.write(f"Country = {country}")
        # Load the label encoder and transform the country input
        le = joblib.load('label_encoder.pkl')  # Load the label encoder
        country_encoded = le.fit_transform([country])[0]

        # Create a DataFrame with user input features and encoded country
        features = pd.DataFrame([data])
        features['Country_encoded'] = country_encoded

        # Ensure that the input DataFrame contains all 20 features expected by the model
        # You need to adjust this part based on your data
        # Add default values for missing features
        expected_features = [
           "BirthRate", "CO2Emissions", "EnergyUsage", "GDP", "HealthExpGDP",
           "HealthExpCapita", "InfantMortalityRate", "InternetUsage", "LendingInterest",
           "LifeExpectancyFemale", "LifeExpectancyMale", "MobilePhoneUsage",
           "Population0to14", "Population15to64", "Populationmorethan65",
           "PopulationTotal", "PopulationUrban", "TourismInbound", "TourismOutbound",
]
        for feature in expected_features:
            if feature not in features.columns:
                features[feature] = 0  # Set default value (change as needed)

        # Load the trained model
        clf = joblib.load('trained_model.pkl')

        # Make predictions
        prediction = clf.predict(features)
        prediction =prediction + 1
        cluster_names = {
            1: "Environmental Concerns",
            2: "Challenging Business Environment",
            3: "Developed Country",  
            4: "Emerging Markets"
        }

        # Display cluster name based on the predicted value
        if prediction[0] in cluster_names:
            predicted_cluster = cluster_names[prediction[0]]
            st.subheader("Prediction")
            st.write(predicted_cluster)
        else:
            st.subheader("Prediction")
            st.write("Unknown")

if __name__ == "__main__":
    main()
