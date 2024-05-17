from pickle import load

import streamlit as st


#Loading the model
model_path = "/models/tree_classifier_crit-gini_maxdepth-5_minleaf-1_minsplit-2_8.sav"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)



class_dict = {
    "0": "Non diabetic",
    "1": "Diabetic",
}

#Configuration:
def main():
    st.title("Diabetes - Model prediction")

    st.write("This predictor uses a Decision tree optimized model with a 74% accuracy in predicting if someone is diabetic depending on several parameters.")

    st.write("Please fill in the blanks with your information:")
    
    number1 = st.number_input("Pregnancies", value=None, placeholder="Type a number...")
    st.divider()
    number2 = st.number_input("Glucose", value=None, placeholder="Type a number...")
    st.divider()
    number3 = st.number_input("Skin Thickness", value=None, placeholder="Type a number...")
    st.divider()
    number4 = st.number_input("Insulin", value=None, placeholder="Type a number...")
    st.divider()
    number5 = st.number_input("BMI", value=None, placeholder="Type a number...")
    st.divider()
    number6 = st.number_input("Diabetes Pedigree Function", value=None, placeholder="Type a number...")
    st.divider()
    number7 = st.number_input("Age", value=None, placeholder="Type a number...")

    if st.button("Predict"):

        
        prediction = str(model.predict([[number1, number2, number3, number4, number5, number6, number7]])[0])
        pred_class = class_dict[prediction]
        st.write("Prediction:", pred_class)

if __name__ == '__main__':
    main()

