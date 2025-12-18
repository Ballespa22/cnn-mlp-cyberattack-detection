import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import os

# Verificar versión de TensorFlow
print(f"[INFO] Versión de TensorFlow: {tf.__version__}")

# Configurar semillas
np.random.seed(42)
tf.random.set_seed(42)
print("[INFO] Semillas configuradas.")

# Limpiar sesión
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
print("[INFO] Sesión de Keras y grafo TensorFlow limpiados.")

# Configurar dispositivo
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("[INFO] Usando GPU:", physical_devices)
else:
    print("[INFO] Usando CPU:", physical_devices)

# Crear directorio de salida
output_dir = 'CNN_output'
os.makedirs(output_dir, exist_ok=True)

# Cargar los datos preprocesados
print("[INFO] Cargando datos preprocesados...")
start_time = time.time()
X_train = np.load('X_train_cnn_v7.npy')
X_val = np.load('X_val_cnn_v7.npy')
X_test = np.load('X_test_cnn_v7.npy')
y_train = np.load('y_train_cnn_v7.npy')
y_val = np.load('y_val_cnn_v7.npy')
y_test = np.load('y_test_cnn_v7.npy')
load_time = time.time() - start_time
print(f"[INFO] Datos cargados en {load_time:.2f} segundos.")

# Data Augmentation
print("[INFO] Aplicando data augmentation...")
# Convertir X_train a 2D para SMOTE (aplanar temporalmente)
X_train_2d = X_train.reshape(X_train.shape[0], -1)
smote = SMOTE(random_state=42)
X_train_2d_res, y_train_res = smote.fit_resample(X_train_2d, y_train)

# Reconstruir X_train a forma 3D (añadir canal)
X_train_res = X_train_2d_res.reshape(-1, 8, 8, 1)

# Transformaciones básicas (rotaciones y reflexiones)
def augment_data(X, y):
    X_aug = [X]
    y_aug = [y]
    for _ in range(3):  # Rotaciones 90°, 180°, 270°
        X_rot = np.rot90(X, k=(_ + 1), axes=(1, 2))
        X_aug.append(X_rot)
        y_aug.append(y)
    X_flip_h = np.flip(X, axis=1)  # Reflejo horizontal
    X_aug.append(X_flip_h)
    y_aug.append(y)
    X_flip_v = np.flip(X, axis=2)  # Reflejo vertical
    X_aug.append(X_flip_v)
    y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)

X_train_aug, y_train_aug = augment_data(X_train_res, y_train_res)

# Asegurarse de que los datos tengan la forma correcta para CNN (ya está en 8x8x1)
if len(X_train_aug.shape) == 3:  # Si es (samples, 8, 8), añadir canal
    X_train_aug = X_train_aug[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

# Normalizar los datos (si no están normalizados)
X_train_aug = X_train_aug.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convertir etiquetas a formato one-hot
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train_aug, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Calcular class weights (normalizados por frecuencia máxima, límite 6)
class_counts = np.bincount(y_train_aug)
max_count = np.max(class_counts)
class_weights = {i: min(6.0, max_count / count) for i, count in enumerate(class_counts) if count > 0}
print("[INFO] Pesos por clase (normalizados por frecuencia máxima, límite 6):", class_weights)

# Definir la arquitectura de la CNN (ajustada para heatmaps 8x8 con capa adicional)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1), padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.6),
    Dense(128, activation='relu'),
    Dropout(0.6),
    Dense(num_classes, activation='softmax')
])

# Compilar el modelo con un learning rate más bajo
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Callbacks para early stopping y log
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
csv_logger = CSVLogger(os.path.join(output_dir, 'training_log.csv'), append=True, separator=',')
callbacks = [early_stopping, csv_logger]

# Entrenar el modelo con class_weight y batch_size mayor
total_epochs = 50
print(f"[INFO] Iniciando entrenamiento para {total_epochs} épocas...")
start_time = time.time()
history = model.fit(X_train_aug, y_train_cat,
                    batch_size=128,
                    epochs=total_epochs,
                    validation_data=(X_val, y_val_cat),
                    verbose=1,
                    callbacks=callbacks,
                    class_weight=class_weights)

train_time = time.time() - start_time
print(f"[INFO] Entrenamiento completado en {train_time:.2f} segundos. Ejecutadas {len(history.epoch) + 1} épocas.")

# Evaluar el modelo
start_time = time.time()
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
eval_time = time.time() - start_time
print(f"\nPrecisión en el conjunto de prueba: {test_accuracy:.4f} (evaluado en {eval_time:.2f} segundos)")

# Predecir etiquetas para calcular F1-score
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_cat, axis=1)

# Calcular F1-score por clase
f1_scores = f1_score(y_test_classes, y_pred_classes, average=None)
class_report = classification_report(y_test_classes, y_pred_classes, zero_division=0)

print("\nF1-score por clase:")
for i, f1 in enumerate(f1_scores):
    print(f"Clase {i}: {f1:.4f}")
print("\nReporte completo de clasificación:")
print(class_report)

# Calcular AUC-ROC por clase (one-vs-rest)
auc_roc_scores = roc_auc_score(y_test_cat, y_pred, multi_class='ovr', average=None)
auc_roc_macro = roc_auc_score(y_test_cat, y_pred, multi_class='ovr', average='macro')

print("\nAUC-ROC por clase:")
for i, auc in enumerate(auc_roc_scores):
    print(f"Clase {i}: {auc:.4f}")
print(f"AUC-ROC macro promedio: {auc_roc_macro:.4f}")

# Graficar la precisión y pérdida
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cnn_training_history_v7.png'))
plt.close()

# Generar y guardar matriz de confusión
cm = tf.math.confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_v7.png'))
plt.close()

# Generar y guardar heatmap de ejemplo
sample_data = X_test[0].reshape(8, 8)  # Tomar el primer ejemplo de test
plt.figure(figsize=(6, 5))
sns.heatmap(sample_data, annot=True, fmt='.2f', cmap='viridis')
plt.title('Ejemplo de Heatmap de Datos de Entrada')
plt.savefig(os.path.join(output_dir, 'example_heatmap_v7.png'))
plt.close()

# Guardar el modelo
model.save(os.path.join(output_dir, 'cnn_model_v7.keras'))
print("[INFO] Modelo guardado como cnn_model_v7.keras en CNN_output")