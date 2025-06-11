import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

class SolarPanelDataProcessor:
    def __init__(self, data_dir, img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.classes = ['Clean', 'Dusty', 'Bird-Drop', 'Electrical-Damage', 'Physical-Damage', 'Snow-Covered']
        self.valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        self.class_distribution = {}

    def load_data(self):
        images, labels = [], []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            self.class_distribution[class_name] = 0
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(self.valid_exts):
                    continue
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                img = preprocess_input(img)
                images.append(img)
                labels.append(class_idx)
                self.class_distribution[class_name] += 1
        return np.array(images), np.array(labels)

    def build_model(self, num_classes):
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_and_save_model(self, save_path='solar_panel_classifier.h5'):
        X, y = self.load_data()
        y_cat = to_categorical(y, num_classes=len(self.classes))

        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=np.argmax(y_train, axis=1), random_state=42)

        train_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
                                       height_shift_range=0.1, horizontal_flip=True)
        val_gen = ImageDataGenerator()
        test_gen = ImageDataGenerator()

        model = self.build_model(num_classes=len(self.classes))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True)
        ]

        model.fit(train_gen.flow(X_train, y_train, batch_size=32),
                  validation_data=val_gen.flow(X_val, y_val),
                  epochs=20, callbacks=callbacks)

        print(f"✅ Model saved to: {save_path}")

if __name__ == "__main__":
    processor = SolarPanelDataProcessor(data_dir='Solar_Panel_Dataset')
    processor.train_and_save_model()
