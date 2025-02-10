import streamlit as st
import pickle
import numpy as np
import joblib



def load_model():
    with open("lass_reg_model.jb", "rb") as file:
        model = joblib.load(file)
    return model

def main():
    st.title("Car Price Predictor")
    st.write("Enter the details of the car to estimate its selling price.")
    
    # User inputs
    year = st.number_input("Year", min_value=2000, max_value=2025, step=1)
    present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, format="%.2f")
    kms_driven = st.number_input("Kilometers Driven", min_value=0, step=500)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner Type", [0, 1, 2, 3])
    
    # Encoding categorical variables
    fuel_dict = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    seller_dict = {"Dealer": 0, "Individual": 1}
    transmission_dict = {"Manual": 0, "Automatic": 1}
    
    fuel_type_encoded = fuel_dict[fuel_type]
    seller_type_encoded = seller_dict[seller_type]
    transmission_encoded = transmission_dict[transmission]
    
    # Load model
    model = load_model()
    
    # Predict button
    if st.button("Predict Price"):
        input_data = np.array([[year, present_price, kms_driven, fuel_type_encoded, seller_type_encoded, transmission_encoded, owner]])
        prediction = model.predict(input_data)
        st.success(f"Estimated Selling Price: ₹{prediction[0]:.2f} Lakhs")

if __name__ == "__main__":
    main()
