import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("alina_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load OneHotEncoder and StandardScaler
with open("onehot_encoder.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)  # Corrected the unmatched parenthesis here

# Title of the application
st.title("Titanic Survival Prediction")

st.write("""And the symbiol of the power of humanity...""") 
st.write(""" A full-metal challenge to the natutre...""") 
st.write("""It has no fear...""") 
st.write("""It is colosssal, magnificent and invincible...""") 
st.write(""" --William Thomas Stead.""")




st.image("titanic_ship.jpg", use_column_width=True)


 #Problem Presentation
st.header("Project Overview")
st.write("""
    The Titanic, one of the most famous maritime disasters in history, sank on April 15, 1912, after colliding with an iceberg during its maiden voyage from Southampton to New York City. Out of the 2,224 passengers and crew aboard, only 710 survived, leaving over 1,500 lives lost.

Our Titanic Survival Prediction Model leverages machine learning to analyze passenger data and predict whether an individual would have survived the tragedy. Built using a Gradient Boosting algorithm and trained on the well-known Titanic dataset from Kaggle, the model considers features like age, gender, passenger class, and ticket information to make its predictions.

This tool not only showcases the power of machine learning but also highlights the real-life factors that influenced survival during the Titanic disaster. By using historical data, the model provides insights into the impact of social and demographic factors on survival outcomes.
Fans of the Titanic movie and history can use a model to predict if they would have survived the real Titanic catastrophe.""")


st.header("Enter passenger details to predict whether they would survive.")

st.write("To choose your class, look at the information below and decide based on how much you want to spend on a 6-7 day trip.")

# Text to display
text = """
**Passengers 1st class:**
- Ticket price in 1912: $500
- Today's equivalent: $15,950

**Passengers 2nd class:**
- Ticket price in 1912: $30-50
- Today's equivalent: $957-1,595

**Passengers 3rd class:**
- Ticket price in 1912: $10-20
- Today's equivalent: $319-638
"""

# Display the text
st.markdown(text)

pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd):", [1, 2, 3])
sex = st.selectbox("Gender:", ["Male", "Female"])
age = st.slider("Age (years):", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard:", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard:", min_value=0, max_value=10, value=0)


st.write("Comeback to fisrt step and put approximate amount that you are redy to spend on ticket and think about extra charges for facilities usage.")

fare = st.slider("Ticket Fare ($):", min_value=0.0, max_value=500.0, value=30.0)

# User inputs
st.write(""" Choose the closest destination to your location to find where you can start traveling on the Titanic.""")
st.write(""" C: Cherbourg, France""")
st.write(""" Q: Queenstown, Ireland""")
st.write(""" S: Southampton, England""")

embarked = st.selectbox("Port of Embarkation:", ["C", "Q", "S"])

if st.button("Predict Survival"):
  #  try: 
    # Prepare input data
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })



    # Preprocess categorical features
    cat_features = input_data[["Sex", "Embarked"]]
    cat_transformed = ohe.transform(cat_features)  # Apply the same OneHotEncoder

    cat_transformed = pd.DataFrame(cat_transformed,
                                   columns=ohe.get_feature_names_out(),  # Correct usage
                                   index=cat_features.index)


    # Preprocess numerical features
    num_features = input_data[["Age", "Fare"]]
    num_transformed = scaler.transform(num_features)
    num_transformed = pd.DataFrame(num_transformed,
                                   columns=num_features.columns,  # Use the original column names
                                   index=num_features.index)


    # Combine processed features
    input_full = pd.concat([num_transformed, cat_transformed, input_data.drop(columns=["Sex", "Embarked", "Age", "Fare"])], axis=1)

    #st.write("Combined input shape:", input_full.shape)
    #st.write(input_full)
    #print(model.feature_importances_)

    # Align with training columns
   # st.write("Model expected columns:", model.feature_names_in_)
    input_full = input_full.reindex(columns=model.feature_names_in_, fill_value=0)

    #st.write("Aligned input shape:", input_full.shape)
    #st.write(input_full)

    # Make prediction
    prediction = model.predict(input_full)
    survival = "Survived" if prediction[0] == 1 else "Did Not Survive"

    # Display the result
    st.write(f"Prediction: **{survival}**")



    st.write(""" More than a hundred years have passed since the Titanic sank, and much has seemingly changed: transatlantic flights, space exploration, nuclear energy, satellite phones. We still desire unlimited prosperity, maximum comfort, and absolute reliability, yet we remain absolutely defenseless against earthquakes, tsunamis, typhoons, and floating ice.

There are endless 'what ifs' and 'who is to blame,' and meanwhile, at the bottom of the Atlantic, the symbol of humanity's power is slowly crumbling to dust—a full-metal challenge to nature—the Unsinkable Titanic""")



    #except Exception as e:
    #st.error(f"An error occurred: {e}")


    #except Exception as e:
       # st.error(f"An error occurred: {e}")