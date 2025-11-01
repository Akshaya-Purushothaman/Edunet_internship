import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Synthetic data
num_samples = 100
IMG_SIZE = (128, 128, 3)
X_temp = np.random.rand(num_samples, *IMG_SIZE)
X_norm = np.random.rand(num_samples, *IMG_SIZE)
y = np.random.randint(0, 3, size=(num_samples,))
y = to_categorical(y, 3)

# Create CNN branches properly with Input layers
def create_branch(name):
    inp = layers.Input(shape=IMG_SIZE, name=f"{name}_input")
    x = layers.Conv2D(32, (3,3), activation='relu')(inp)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Flatten()(x)
    return inp, x

# Two branches: temperature and normal
temp_inp, temp_out = create_branch("temp")
norm_inp, norm_out = create_branch("norm")

# Combine both features
combined = layers.concatenate([temp_out, norm_out])
x = layers.Dense(64, activation='relu')(combined)
x = layers.Dropout(0.3)(x)
output = layers.Dense(3, activation='softmax')(x)

# Define final model
model = models.Model(inputs=[temp_inp, norm_inp], outputs=output)
model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 Training demo CNN model for UHI detection...")
model.fit([X_temp, X_norm], y, epochs=5, batch_size=8, validation_split=0.2)

model.save('uhi_model.h5')
print("✅ Model saved as uhi_model.h5")

