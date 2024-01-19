import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Path to the dataset directory
dataset_path = 'dataset/'

# Define classes based on your dataset subdirectories
classes = [
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite'
]

# Split the data into training and testing sets
train_data, test_data = train_test_split(classes, test_size=0.2, random_state=42)

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Flow from directory for training data
train_generator = datagen.flow_from_directory(
    directory=dataset_path,
    classes=train_data,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Flow from directory for testing data
test_generator = datagen.flow_from_directory(
    directory=dataset_path,
    classes=test_data,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add new classification layers
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=2,
    validation_data=test_generator
)

# Save the trained model
model.save('retrained_model.h5')
