import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf
#import cv2  # We might need this, but let's try with pure NumPy first to keep it simple!

# 1. LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('digit_model.keras')

model = load_model()

# --- THE NEW MAGIC FUNCTION ---
def center_image(img):
    """
    Crops the digit, RESIZES it to 20x20, and centers it in a 28x28 frame.
    """
    # Convert to numpy array to find the bounding box
    img_array = np.array(img)
    
    # 1. Find the bounding box
    coords = np.argwhere(img_array > 0)
    if coords.shape[0] == 0:
        return img
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 2. Crop the image
    cropped_array = img_array[y_min:y_max+1, x_min:x_max+1]
    
    # --- NEW PART STARTS HERE ---
    
    # 3. Convert crop back to PIL Image to resize it easily
    cropped_img = Image.fromarray(cropped_array)
    
    # 4. Resize while keeping aspect ratio
    # We want the largest side to be 20 pixels (leaving 4px padding on all sides)
    w, h = cropped_img.size
    
    if w > h:
        new_w = 20
        new_h = int(h * (20 / w))
    else:
        new_h = 20
        new_w = int(w * (20 / h))
        
    resized_img = cropped_img.resize((new_w, new_h))
    
    # 5. Paste into a 28x28 black box
    final_img = Image.new('L', (28, 28), color=0) # 'L' means grayscale
    
    # Calculate centering coordinates
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    
    final_img.paste(resized_img, (paste_x, paste_y))
    
    return final_img


st.set_page_config(page_title="Digit Recognizer", page_icon="🔢")

if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

def clear_canvas():
    st.session_state.reset_key += 1

st.sidebar.header("🎨 Settings")
stroke_width = 30
st.sidebar.markdown("**Instructions:** Draw a digit (0-9).")

st.title("🧠 Deep Learning Digit Recognizer")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Your Drawing")
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        display_toolbar=False,
        key=f"canvas_{st.session_state.reset_key}",
    )
    if st.button("🗑️ Clear Canvas"):
        clear_canvas()
        st.rerun()

with col2:
    st.subheader("Prediction")
    
    if st.button('🔮 Predict', use_container_width=True):
        if canvas_result.image_data is not None:
            
            # --- 1. PRE-PROCESSING ---
            # Get raw image
            raw_img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
            
            # *** APPLY THE NEW CENTERING MAGIC ***
            centered_img = center_image(raw_img)
            
            # Resize to 8x8 (Model's size)
            final_img = centered_img.resize((8, 8))
            
            # --- 2. PREPARE FOR AI ---
            pixel_array = np.array(final_img)
            pixel_array = pixel_array / 255.0  # Normalize 0.0-1.0
            img_input = pixel_array.reshape(1, 8, 8, 1)
            
            # --- 3. PREDICT ---
            prediction = model.predict(img_input)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            st.markdown(f"# **{predicted_digit}**")
            st.caption(f"Confidence: {confidence:.2%}")
            
            # --- DEBUG VIEW ---
            with st.expander("Debug View"):
                st.write("Centered & Resized (What AI Sees):")
                st.image(final_img.resize((150, 150), resample=Image.NEAREST))
                st.bar_chart(prediction[0])
        else:
            st.warning("Draw something first!")