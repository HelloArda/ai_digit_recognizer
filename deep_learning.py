import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models
from sklearn import datasets # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import joblib # type: ignore

# 1. LOAD DATA
# We use the same 8x8 data, but Neural Networks prefer numbers 0-1
digits = datasets.load_digits()
X = digits.images  # We load .images to keep the 8x8 grid structure!
y = digits.target

# Normalize (Scale 0-16 down to 0.0-1.0)
X = X / 16.0

# 2. RESHAPE FOR CNN
# CNNs expect (Rows, Cols, ColorChannels). 
# We have 1 color (Grayscale). So we reshape 8x8 -> 8x8x1
# The '-1' tells Python: "You figure out how many images there are, just make the rest match."
X = X.reshape(-1, 8, 8, 1)

# 3. SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cnn = models.Sequential()

cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)))
cnn.add(layers.MaxPooling2D((2, 2)))

# 3. Flatten (3D -> 1D)
cnn.add(layers.Flatten())

# 4. The Output Layer
# 64 neurons to process the features, then 10 neurons for the final answer
cnn.add(layers.Dense(64, activation='relu'))
cnn.add(layers.Dense(10, activation='softmax'))

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=5)

cnn.save('digit_model.keras')