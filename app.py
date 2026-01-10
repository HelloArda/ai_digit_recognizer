import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import joblib

model = joblib.load("digits_model.pkl")

st.set_page_config(page_title="Digit Recognizer", page_icon="🔢")

if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

def clear_canvas():
    st.session_state.reset_key += 1

st.sidebar.header("🎨 Settings")
stroke_width = st.sidebar.slider("Stroke Width", 10, 30, 20)
st.sidebar.markdown("**Instructions:** Draw a digit (0-9) big in the center.")

st.title("🔢 AI Digit Recognizer")

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
           
            img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L').resize((8, 8))
            pixel_array = (np.array(img) / 255.0) * 16.0
            prediction = model.predict(pixel_array.flatten().reshape(1, -1))
            
            st.markdown(f"# **{prediction[0]}**")
            
            with st.expander("Debug View"):
                st.image(img.resize((150, 150), resample=Image.NEAREST))
        else:
            st.warning("Draw something first!")