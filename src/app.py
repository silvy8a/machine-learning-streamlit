# Define the model path
model_path = "../models/tree_classifier_crit-gini_maxdepth-5_minleaf-1_minsplit-2_8.pkl"

# Check if the model file exists
if not os.path.isfile(model_path):
    st.error(f"Model file not found at path: {model_path}")
    st.stop()

# Loading the model
try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

class_dict = {
    "0": "Non diabetic",
    "1": "Diabetic",
}

# Application Configuration
def main():
    st.title("Diabetes Prediction Model")

    st.write("This predictor utilizes a Decision Tree model optimized with an accuracy of 74% to predict diabetes based on various health parameters. Please provide the following information:")

    # Input fields with user guidance
    number1 = st.number_input("Number of Pregnancies", min_value=0, step=1, format="%d", help="Enter the number of times you've been pregnant.")
    st.divider()
    number2 = st.number_input("Glucose Level (mg/dL)", min_value=0, help="Enter your glucose level in mg/dL.")
    st.divider()
    number3 = st.number_input("Skin Thickness (mm)", min_value=0, help="Enter the skin thickness in mm.")
    st.divider()
    number4 = st.number_input("Insulin Level (IU/mL)", min_value=0, help="Enter your insulin level in IU/mL.")
    st.divider()
    number5 = st.number_input("Body Mass Index (BMI)", min_value=0.0, format="%.2f", help="Enter your BMI.")
    st.divider()
    number6 = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f", help="Enter your diabetes pedigree function value.")
    st.divider()
    number7 = st.number_input("Age (years)", min_value=0, step=1, format="%d", help="Enter your age in years.")

    # Predict button
    if st.button("Predict"):
        # Validate inputs before making a prediction
        if any(v is None for v in [number1, number2, number3, number4, number5, number6, number7]):
            st.error("Please fill in all the fields with valid numbers.")
        else:
            try:
                prediction = str(model.predict([[number1, number2, number3, number4, number5, number6, number7]])[0])
                pred_class = class_dict[prediction]
                st.write("Prediction:", pred_class)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
