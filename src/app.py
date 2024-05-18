import os
import pickle
import streamlit as st

# Adjust the model directory path based on the current working directory
model_dir = "../models"
model_filename = "tree_classifier_crit-gini_maxdepth-5_minleaf-1_minsplit-2_8.pkl"
model_path = os.path.join(model_dir, model_filename)

# Debugging: print current working directory and directory contents
current_working_directory = os.getcwd()
st.write(f"Current working directory: {current_working_directory}")

# List contents of the parent directory to locate the models directory
parent_dir = os.path.abspath(os.path.join(current_working_directory, os.pardir))
st.write(f"Parent directory: {parent_dir}")
try:
    parent_dir_contents = os.listdir(parent_dir)
    st.write("Parent directory contents:", parent_dir_contents)
except FileNotFoundError as e:
    st.error(f"Parent directory not found: {parent_dir}")
    st.stop()

# Check if the model directory exists and list its contents
if not os.path.isdir(model_dir):
    st.error(f"Model directory not found: {model_dir}")
    st.stop()
else:
    st.write(f"Model directory contents: {os.listdir(model_dir)}")

# Check if the model file exists
if not os.path.isfile(model_path):
    st.error(f"Model file not found at path: {model_path}")
    st.stop()

# Loading the model
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Rest of your Streamlit app...
def main():
    st.title("Diabetes Prediction Model")
    
    st.write("This predictor uses a Decision Tree optimized model with a 74% accuracy in predicting if someone is diabetic depending on several parameters.")
    st.write("Please fill in the blanks with your information:")
    
    number1 = st.number_input("Pregnancies", value=0, min_value=0, step=1)
    st.divider()
    number2 = st.number_input("Glucose", value=0, min_value=0, step=1)
    st.divider()
    number3 = st.number_input("Skin Thickness", value=0, min_value=0, step=1)
    st.divider()
    number4 = st.number_input("Insulin", value=0, min_value=0, step=1)
    st.divider()
    number5 = st.number_input("BMI", value=0.0, min_value=0.0, format="%.2f")
    st.divider()
    number6 = st.number_input("Diabetes Pedigree Function", value=0.0, min_value=0.0, format="%.3f")
    st.divider()
    number7 = st.number_input("Age", value=0, min_value=0, step=1)
    
    if st.button("Predict"):
        try:
            prediction = model.predict([[number1, number2, number3, number4, number5, number6, number7]])[0]
            class_dict = {0: "Non diabetic", 1: "Diabetic"}
            pred_class = class_dict[prediction]
            st.write("Prediction:", pred_class)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()


