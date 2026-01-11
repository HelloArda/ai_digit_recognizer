# 🔢 AI Digit Recognizer

A machine learning web application that recognizes handwritten digits (0-9) in real-time. Built with Python, Scikit-Learn, and Streamlit.

## 🚀 Demo
*(You can add a screenshot of your app here later!)*

## 🧠 How it Works
1.  **The Model:** Uses a Support Vector Machine (SVM) trained on the MNIST dataset (8x8 pixel version).
2.  **The Interface:** A Streamlit canvas allows users to draw digits directly on the screen.
3.  **Processing:** The app resizes the drawing to 8x8 grayscale pixels to match the model's training data.

## 🛠️ Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/ai-digit-recognizer.git](https://github.com/YOUR_USERNAME/ai-digit-recognizer.git)
    cd ai-digit-recognizer
    ```

2.  **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## 💻 Tech Stack
* **Language:** Python 3.9+
* **ML Library:** Scikit-Learn (SVM Classifier)
* **Web Framework:** Streamlit
* **Image Processing:** NumPy, PIL
