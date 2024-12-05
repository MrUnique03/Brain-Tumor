import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model_path = '/workspaces/codespaces-blank/Tumor_model/brain_tumor_cnn_model.h5'
model = tf.keras.models.load_model(model_path)

# Define the classes for prediction
CLASS_TYPES = ['Normal', 'Glioma', 'Meningioma', 'Pituitary']

# Section 1: App Logic

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((150, 150))  # Resize the image
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Home page content
def home_page():
    st.title("Welcome to the Brain Tumor Detection System! 🧠🔬")

     # Add an image to the homepage
    image = Image.open('path_to_image.jpg')  # Specify your image path here
    st.image(image, caption="Brain Tumor Detection System", use_column_width=True)



    # Adding the mission and how it works section
    st.subheader("Our Mission")
    st.write(
        """
        Our mission is to help in detecting brain tumors efficiently and accurately. 
        Upload an MRI image of the brain, and our system will analyze it to identify potential tumor types. 
        Together, let's take a step towards early detection and better healthcare outcomes!
        """
    )

    st.subheader("How It Works")
    st.write(
        """
        1. **Upload Image:** Go to the **Tumor Recognition** page and upload an image of a brain MRI with suspected tumors.
        2. **Analysis:** Our system processes the image using advanced deep learning algorithms to detect any tumor and classify it.
        3. **Results:** View the predicted tumor type along with the confidence score for further action.
        """
    )

    st.subheader("Why Choose Us?")
    st.write(
        """
        - **Accuracy:** Our system boasts a high accuracy rate of **94%** in detecting and classifying brain tumors.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Get results within seconds, enabling timely decisions for medical professionals.
        """
    )

    st.subheader("Get Started")
    st.write(
        """
        Click on the **Tumor Checker** page in the sidebar to upload an image and experience the power of our Brain Tumor Detection System!
        """
    )

# About page content
def about_page():
    st.title("About")
    
    # Mission
    st.subheader("Mission")
    st.write(
        "Our mission is to provide efficient and reliable solutions for detecting brain tumors using state-of-the-art machine learning algorithms."
    )
    
    # Vision
    st.subheader("Vision")
    st.write(
        "We aim to improve the accuracy of medical diagnoses by leveraging AI technologies in the healthcare sector."
    )
    
    # Our Team
    st.subheader("Our Team")
    st.write("Our team consists of professionals dedicated to improving healthcare through innovation.")

    # About Project
    st.subheader("About the Project")
    st.write(
        """
        This project focuses on the detection of brain tumors from MRI images using deep learning techniques. The dataset includes the following number of images:
        - **Number of Normal images**: 3066
        - **Number of Glioma Tumor images**: 6307
        - **Number of Meningioma Tumor images**: 6391
        - **Number of Pituitary Tumor images**: 5908
        
        We trained the model using the data, achieving a **94% accuracy** in tumor classification. The model was trained over 20 epochs, and the results demonstrated significant improvement, with validation accuracy reaching up to **96.72%** by the final epoch.
        
        This project was developed by **Aliyu Kolawale Abdulwahid**, a SIWES intern, and **Rodiyat**, an NYSC member, both of whom interned at **NCAIR**.
        """
    )

# Tumor Checker page content
def tumor_checker_page():
    st.title("Tumor Checker")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=300)

        # Buttons for user control
        col_buttons = st.columns(3)
        run_button = col_buttons[0].button("Run Model", key="run", help="Click to run the model")
        reupload_button = col_buttons[1].button("Reupload Image", key="reupload", help="Click to upload a new image")

        # Action after pressing 'Run Model'
        if run_button:
            st.info("Processing the image...")
            processed_image = preprocess_image(img)  # Preprocess the image
            prediction = model.predict(processed_image)  # Get model prediction
            predicted_class_index = np.argmax(prediction, axis=-1)[0]
            predicted_class = CLASS_TYPES[predicted_class_index]

            # Display result
            st.success(f"Prediction: The model predicts **{predicted_class}**.")
            st.write(f"Confidence: {prediction[0][predicted_class_index]*100:.2f}%")

        # Action after pressing 'Reupload Image'
        if reupload_button:
            st.info("Please upload a new image.")
            uploaded_file = None  # Reset the uploaded file
            st.image([], caption="No image", width=300)  # Clear the image display

# Sidebar layout for navigation
def set_page(page_name):
    """Sets the selected page."""
    st.session_state.page = page_name

# Initialize session state for page selection
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Sidebar buttons to navigate between pages
def sidebar():
    st.sidebar.title("Dashboard")
    st.sidebar.button("Home", on_click=set_page, args=("Home",))
    st.sidebar.button("About", on_click=set_page, args=("About",))
    st.sidebar.button("Tumor Checker", on_click=set_page, args=("Tumor Checker",))

# Section 2: CSS Styling Function
# Section 2: CSS Styling Function
def set_custom_css():
    st.markdown("""
    <style>
        /* General styles for the page */
        body {
            background-color: #FAF6E3;  /* Soft cream background */
            color: #2A3663;  /* Dark blue text */
        }

        /* Sidebar styles */
        .css-1d391kg {
            background-color: #2A1A45;  /* Darker purple background for sidebar */
            color: #FAF6E3;  /* Soft cream text */
            padding: 20px;
        }

        .css-1d391kg h1 {
            color: #FAF6E3;  /* Soft cream text */
            font-size: 24px;
            text-align: center;
            padding: 20px;
        }

        /* Sidebar box for Dashboard */
        .dashboard-box {
            background-color: #1A0D30;  /* Darker background for Dashboard box */
            color: #FAF6E3;  /* Soft cream text */
            padding: 15px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            margin-bottom: 30px;  /* More space below the Dashboard box */
            text-align: center;
        }

        /* Sidebar box for other sections (Home, About, Tumor Checker) */
        .section-box {
            background-color: #B59F78;  /* Medium brown for other sections */
            color: #FAF6E3;  /* Soft cream text */
            padding: 10px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 10px;
            margin-bottom: 20px;  /* More space between section buttons */
        }

        /* Space out sections evenly */
        .css-1d391kg .stSidebar {
            display: flex;
            flex-direction: column;
            justify-content: space-evenly;
            height: 100%;
        }

        /* Ensure all sidebar buttons and sections are styled with hover effects */
        .stSidebar .stRadio div:hover {
            background-color: #2A3663;  /* Dark blue */
            color: #FAF6E3;  /* Soft cream */
        }

        /* Styling for the Home, About, Tumor Checker boxes */
        .stButton button {
            background-color: #2A3663;  /* Dark blue */
            color: #FAF6E3;  /* Soft cream */
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
            transition: background-color 0.3s ease;
            margin-bottom: 15px;  /* Space between buttons */
        }
        .stButton button:hover {
            background-color: #FAF6E3;  /* Soft cream */
            color: #2A3663;  /* Dark blue */
        }

        /* Add padding between the sidebar text and buttons */
        .css-1d391kg .stSidebar .stButton {
            margin-top: 15px;
        }
    </style>
    """, unsafe_allow_html=True)


# Section 3: Main Layout and Flow
def main():
    # Set page layout
    st.set_page_config(page_title="NCAIR Tumor Detection Centre", layout="wide")

    # Apply custom CSS
    set_custom_css()

    # Sidebar takes up 1/4 of the page width
    col1, col2 = st.columns([1, 4])

    # Sidebar content
    with col1:
        sidebar()

    # Main content based on the selected page
    with col2:
        if st.session_state.page == "Home":
            home_page()
        elif st.session_state.page == "About":
            about_page()
        elif st.session_state.page == "Tumor Checker":
            tumor_checker_page()

# Run the main function to start the app
if __name__ == "__main__":
    main()
