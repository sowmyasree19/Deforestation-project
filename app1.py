import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
import cv2
import numpy as np 
from PIL import Image
import io
import hashlib
import hmac

# Set page configuration
st.set_page_config(
    page_title="Forest Analysis App",
    page_icon="🌲",
    layout="wide"
)

if 'users' not in st.session_state:
    st.session_state.users = {}  
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None


def hash_password(password: str, salt: bytes = None) -> tuple[bytes, bytes]:
    """Hash password with random salt using PBKDF2."""
    if salt is None:
        salt = hashlib.sha256(tf.random.normal([1]).numpy().tobytes()).digest()
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return key, salt

def verify_password(password: str, key: bytes, salt: bytes) -> bool:
    """Verify password against stored hash."""
    new_key, _ = hash_password(password, salt)
    return hmac.compare_digest(key, new_key)

def sign_up():
    """Handle user registration."""
    st.subheader("Create New Account")
    
    with st.form("signup_form"):
        new_username = st.text_input("Username").strip()
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        signup_button = st.form_submit_button("Sign Up")
        
        if signup_button:
            if not new_username or not new_password:
                st.error("Please fill in all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif new_username in st.session_state.users:
                st.error("Username already exists.")
            else:
                key, salt = hash_password(new_password)
                st.session_state.users[new_username] = {
                    'key': key,
                    'salt': salt
                }
                st.success("Account created successfully! Please sign in.")
                return True
    return False

def sign_in():
    """Handle user authentication."""
    st.subheader("Sign In")
    
    with st.form("signin_form"):
        username = st.text_input("Username").strip()
        password = st.text_input("Password", type="password")
        signin_button = st.form_submit_button("Sign In")
        
        if signin_button:
            if not username or not password:
                st.error("Please fill in all fields.")
            elif username not in st.session_state.users:
                st.error("Username not found.")
            else:
                user_data = st.session_state.users[username]
                if verify_password(password, user_data['key'], user_data['salt']):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Logged in successfully!")
                    return True
                else:
                    st.error("Incorrect password.")
    return False

def create_classification_model():
    """Recreate the classification model architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(256, 256, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def create_segmentation_model():
    """Recreate the U-Net architecture"""
    img_height, img_width = 256, 256
    
    inputs = Input((img_height, img_width, 3))
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    
    # Decoder
    u2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u2 = concatenate([u2, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    u1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u1 = concatenate([u1, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_models():
    """Load both classification and segmentation models"""
    try:
        # Create models with correct architecture
        classification_model = create_classification_model()
        segmentation_model = create_segmentation_model()
        
        # Load weights
        classification_model.load_weights('sequential.h5')
        segmentation_model.load_weights('forest_segmentation_model.h5')
        
        return classification_model, segmentation_model
    except Exception as e:
        raise Exception(f"Error loading models: {str(e)}")

def preprocess_image(image, img_size=(256, 256)):
    """Preprocess the uploaded image"""
    # Convert PIL Image to numpy array
    image = np.array(image)
    
    # Ensure image is in RGB
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # If RGBA, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize image
    image = cv2.resize(image, img_size)
    
    # Normalize pixel values
    image = image.astype('float32') / 255.0
    
    # Add batch dimension
    return np.expand_dims(image, axis=0)

def post_process_mask(mask):
    """Post-process the segmentation mask"""
    # Remove batch dimension and get first channel
    mask = mask[0, :, :, 0]
    
    # Threshold the mask
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Apply color map for visualization
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
    # Convert to RGB for display
    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
    
    return mask_colored

def create_overlay(original_image, mask):
    """Create a transparent overlay of the mask on the original image"""
    # Ensure images are the same size
    original_image = cv2.resize(original_image, (256, 256))
    
    alpha = 0.6
    overlay = cv2.addWeighted(original_image, alpha, mask, 1-alpha, 0)
    
    return overlay

def main():
    st.title("🌲 Forest Analysis Tool")
    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
        
        with tab1:
            if sign_in():
                st.experimental_rerun()
                
        with tab2:
            if sign_up():
                st.experimental_rerun()
        
        st.stop()  
    
    col1, col2 = st.columns([3,1])
    with col1:
        st.write(f"Welcome, {st.session_state.username}! 👋")
    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.experimental_rerun()
    st.write("Upload an image for forest classification and segmentation analysis")
    # Display TensorFlow version in sidebar
    st.sidebar.write(f"TensorFlow Version: {tf.__version__}")
    
    # Add confidence threshold slider in sidebar
    confidence_threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Adjust this threshold to fine-tune the classification boundary"
    )
    # Load models
    try:
        classification_model, segmentation_model = load_models()
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please check if the model files exist in the specified path and have the correct format.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            
            # Create three columns for display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption='Input Image', use_column_width=True)
            
            # Preprocess image
            processed_image = preprocess_image(image)
            
            # Run both models
            if st.button('Analyze Image'):
                with st.spinner('Processing...'):
                    # Classification prediction
                    classification_pred = classification_model.predict(processed_image, verbose=0)
                    prediction_class = "Forested" if classification_pred[0][0] >= confidence_threshold else "Deforested"
                    confidence = float(classification_pred[0][0]) if classification_pred[0][0] >= confidence_threshold else 1 - float(classification_pred[0][0])
                    
                    # Segmentation prediction
                    segmentation_pred = segmentation_model.predict(processed_image, verbose=0)
                    
                    # Process segmentation mask
                    mask_colored = post_process_mask(segmentation_pred)
                    
                    # Create overlay
                    original_np = np.array(image)
                    overlay = create_overlay(original_np, mask_colored)
                    
                    # Display results
                    with col2:
                        st.subheader("Segmentation Mask")
                        st.image(mask_colored, caption='Forest Segmentation Mask', use_column_width=True)
                    
                    with col3:
                        st.subheader("Overlay View")
                        st.image(overlay, caption='Overlay of Mask on Original', use_column_width=True)
                    
                    # Display classification results
                    st.write("---")
                    st.subheader("Classification Results")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Classification", prediction_class)
                    with col_b:
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    # Display segmentation statistics
                    st.write("---")
                    st.subheader("Segmentation Statistics")
                    mask_binary = (segmentation_pred[0, :, :, 0] > 0.5).astype(np.uint8)
                    forest_percentage = (np.sum(mask_binary) / mask_binary.size) * 100
                    st.metric("Forest Coverage", f"{forest_percentage:.1f}%")
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.error("Please try uploading a different image or check the image format.")

if __name__ == "__main__":
    main()