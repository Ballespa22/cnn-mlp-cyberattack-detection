import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import os

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
    print("[INFO] Usando CPU.", physical_devices)

# Crear directorio de salida
output_dir = 'MLP_outputs'
os.makedirs(output_dir, exist_ok=True)

# Cargar datos
print("[INFO] Cargando datos...")
X_train = np.load('X_train_v7.npy')
X_val = np.load('X_val_v7.npy')
X_test = np.load('X_test_v7.npy')
y_train = np.load('y_train_v7.npy')
y_val = np.load('y_val_v7.npy')
y_test = np.load('y_test_v7.npy')

# Verificar dimensiones
print("[DEBUG] Dimensiones: X_train={}, X_val={}, X_test={}".format(X_train.shape, X_val.shape, X_test.shape))

# Verificar datos
if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
    print("[ERROR] X_train contiene NaN o valores infinitos.")
    exit(1)
print("[DEBUG] X_train sin problemas.")
if np.any(np.isnan(X_val)) or np.any(np.isinf(X_val)):
    print("[ERROR] X_val contiene NaN o valores infinitos.")
    exit(1)
print("[DEBUG] X_val sin problemas.")
if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
    print("[ERROR] X_test contiene NaN o valores infinitos.")
    exit(1)
print("[DEBUG] X_test sin problemas.")

# Escalar características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

# Verificar solapamiento
def hash_row(row):
    row_str = ''.join(str(x) for x in row)
    return hashlib.sha256(row_str.encode('utf-8')).hexdigest()

print("[INFO] Verificando solapamiento entre conjuntos...")
train_hashes = set(hash_row(row) for row in X_train)
val_hashes = set(hash_row(row) for row in X_val)
test_hashes = set(hash_row(row) for row in X_test)
print("[DEBUG] Instancias comunes entre train y val:", len(train_hashes.intersection(val_hashes)))
print("[DEBUG] Instancias comunes entre train y test:", len(train_hashes.intersection(test_hashes)))
print("[DEBUG] Instancias comunes entre val y test:", len(val_hashes.intersection(test_hashes)))

# Cargar encoder
le = joblib.load('label_encoder_multiclass.pkl')
print("[DEBUG] Clases:", le.classes_)
print("[DEBUG] Ejemplo de etiquetas y_train:", y_train[:10])
print("[DEBUG] Distribución y_train:", np.bincount(y_train))
print("[DEBUG] Distribución y_val:", np.bincount(y_val))
print("[DEBUG] Distribución y_test:", np.bincount(y_test))

# Aplicar undersampling y SMOTE
print("[INFO] Aplicando undersampling y SMOTE...")
rus = RandomUnderSampler(sampling_strategy={
    0: 150000,  # Benign
    6: 100000,  # DoS attacks-Hulk
    1: 80000,   # Bot
}, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
print("[DEBUG] Distribución tras undersampling:", np.bincount(y_train_resampled))

smote = SMOTE(sampling_strategy={
    0: 150000,  # Benign
    1: 80000,   # Bot
    2: 75000,   # Brute Force -Web
    3: 75000,   # Brute Force -XSS
    4: 150000,  # DDoS attacks-LOIC-HTTP
    5: 50000,   # DoS attacks-GoldenEye
    6: 100000,  # DoS attacks-Hulk
    7: 50000,   # DoS attacks-SlowHTTPTest
    9: 50000,   # FTP-BruteForce
    10: 50000,  # Infilteration
    12: 50000   # SSH-Bruteforce
}, random_state=42)
try:
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_resampled, y_train_resampled)
    print("[DEBUG] Nueva distribución y_train_resampled:", np.bincount(y_train_resampled))
except Exception as e:
    print("[ERROR] Error en SMOTE:", e)
    exit(1)

# Generar histograma y tabla de distribución de clases
def plot_distribucion_clases(y, class_names, path):
    class_counts = pd.Series(y).value_counts()
    class_labels = class_counts.index
    class_names_sorted = [class_names[i] for i in class_labels]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_names_sorted, y=class_counts.values)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Clase')
    plt.ylabel('Número de Flujos')
    plt.title('Distribución de Clases tras Undersampling y SMOTE')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'class_distribution.png'))
    plt.close()

    pd.DataFrame({'class': class_names_sorted, 'count': class_counts.values}) \
        .to_csv(os.path.join(path, 'class_distribution.csv'), index=False)

print("[INFO] Generando gráfico y tabla de distribución de clases...")
plot_distribucion_clases(y_train_resampled, le.classes_, output_dir)

# Calcular pesos de clase dinámicos
class_counts = np.bincount(y_train_resampled)
total_samples = len(y_train_resampled)
class_weights = {i: min(total_samples / (len(le.classes_) * count), 10.0) if count > 0 else 1.0 
                 for i, count in enumerate(class_counts)}
normalized_weights = {k: v / sum(class_weights.values()) for k, v in class_weights.items()}
print("[DEBUG] Pesos de clase normalizados:", normalized_weights)

# Definir Focal Loss
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)  # Evitar log(0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_factor = tf.math.pow(1.0 - y_pred, self.gamma)
        focal_loss = self.alpha * focal_factor * cross_entropy
        return tf.reduce_mean(focal_loss, axis=-1)

# Construir MLP
print("[INFO] Construyendo MLP...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(len(le.classes_), activation='softmax')
])
print("[DEBUG] Pesos iniciales primera capa:", model.layers[0].get_weights()[0][0][:10])
model.summary()

# Compilar
print("[INFO] Compilando...")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss=FocalLoss(alpha=0.25, gamma=2.0), metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
print("[INFO] Callbacks configurados: EarlyStopping, ReduceLROnPlateau")

# Convertir etiquetas a one-hot encoding
y_train_resampled = tf.keras.utils.to_categorical(y_train_resampled, num_classes=len(le.classes_))
y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(le.classes_))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(le.classes_))

# Entrenar
print("[INFO] Entrenando...")
history = model.fit(
    X_train_resampled, y_train_resampled,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr]
)

# Depuración
print("[DEBUG] Pérdida/accuracy primera época (train):", history.history['loss'][0], history.history['accuracy'][0])
print("[DEBUG] Pérdida/accuracy primera época (val):", history.history['val_loss'][0], history.history['val_accuracy'][0])

# Guardar modelo
model.save(os.path.join(output_dir, 'mlp_model_optimized_smote_v12.keras'))
print("[INFO] Modelo guardado.")

# Graficar
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mlp_training_history_optimized_smote_v12.png'))
plt.close()
print("[INFO] Gráfico guardado.")

# Evaluar
print("[INFO] Evaluando...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
print("[DEBUG] Forma de y_pred:", y_pred.shape)

# Obtener clases únicas y ajustar target_names
unique_classes = np.unique(y_test_classes)
target_names = [le.classes_[i] for i in unique_classes]

# Reporte
print("\nReporte de Clasificación:")
print(classification_report(y_test_classes, y_pred_classes, labels=unique_classes, target_names=target_names, zero_division=0))

# AUC-ROC
auc_roc = roc_auc_score(y_test, y_pred, multi_class='ovr', labels=range(len(le.classes_)))
print("[INFO] AUC-ROC:", auc_roc)

# Matriz de confusión
cm = tf.math.confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_optimized_smote_v12.png'))
plt.close()
print("[INFO] Matriz de confusión guardada.")

print("[INFO] Ejecución completada.")