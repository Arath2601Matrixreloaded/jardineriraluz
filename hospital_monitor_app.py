"""
Sistema de Monitoreo Acústico Hospitalario - Versión con Entrenamiento
Backend (Flask) + Frontend (HTML/JS) + Deep Learning (TensorFlow) + Training

Autor: Sistema de Monitoreo Hospitalario
Fecha: 2024
"""

import os
import json
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# ============================================================================
# CONFIGURACIÓN DE FLASK
# ============================================================================

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATASET_FOLDER'] = 'dataset'
app.config['MODEL_FOLDER'] = 'models'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'm4a', 'flac'}

# Crear carpetas necesarias
for folder in [app.config['UPLOAD_FOLDER'], app.config['DATASET_FOLDER'], app.config['MODEL_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Crear subcarpetas para cada categoría
CATEGORIES = ['tos', 'grito_dolor', 'alarma_medica']
for cat in CATEGORIES:
    os.makedirs(os.path.join(app.config['DATASET_FOLDER'], cat), exist_ok=True)

# ============================================================================
# CLASE DE MONITOREO ACÚSTICO MEJORADA
# ============================================================================

class HospitalAudioMonitor:
    """Sistema de monitoreo acústico hospitalario usando Deep Learning"""
    
    def __init__(self, sample_rate=22050, duration=3, n_mfcc=40):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.model = None
        self.label_encoder = None
        self.categories = ['tos', 'grito_dolor', 'alarma_medica']
        self.category_names = {
            'tos': 'Tos',
            'grito_dolor': 'Grito de Dolor',
            'alarma_medica': 'Alarma Médica'
        }
        
        # Intentar cargar modelo existente
        self.load_model_if_exists()
    
    def load_model_if_exists(self):
        """Carga modelo si existe"""
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'hospital_model.h5')
        encoder_path = os.path.join(app.config['MODEL_FOLDER'], 'label_encoder.pkl')
        
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            try:
                self.model = keras.models.load_model(model_path)
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print("Modelo pre-entrenado cargado correctamente.")
                return True
            except Exception as e:
                print(f"Error al cargar modelo: {e}")
        
        print("No se encontró modelo entrenado. Usa el modo simulación o entrena un modelo.")
        return False
    
    def load_audio(self, file_path):
        """Carga un archivo de audio"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Padding o truncado para tamaño fijo
            target_length = self.sample_rate * self.duration
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            return audio, sr
        except Exception as e:
            print(f"Error al cargar audio: {str(e)}")
            return None, None
    
    def extract_features(self, audio, sr):
        """Extrae características acústicas del audio"""
        features = {}
        
        # MFCC - 40 coeficientes
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        features['mfcc'] = mfcc
        
        # Espectrograma Mel
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_spectrogram'] = mel_spec_db
        
        # Características adicionales
        features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
        features['rms_energy'] = float(np.mean(librosa.feature.rms(y=audio)))
        # Ancho de banda y planitud espectral para distinguir alarmas (tonales) de ruidos
        try:
            features['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
        except Exception:
            features['spectral_bandwidth'] = 0.0
        try:
            features['spectral_flatness'] = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
        except Exception:
            features['spectral_flatness'] = 0.0
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        
        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = float(tempo)
        except:
            features['tempo'] = 0.0
        
        return features
    
    def prepare_training_data(self):
        """Prepara datos de entrenamiento desde el dataset"""
        X = []
        y = []
        
        print("\n📊 Preparando dataset para entrenamiento...")
        
        for category in self.categories:
            cat_path = os.path.join(app.config['DATASET_FOLDER'], category)
            audio_files = [f for f in os.listdir(cat_path) if f.endswith(tuple(['.'+ext for ext in app.config['ALLOWED_EXTENSIONS']]))]
            
            print(f"   {self.category_names[category]}: {len(audio_files)} archivos")
            
            for audio_file in audio_files:
                file_path = os.path.join(cat_path, audio_file)
                audio, sr = self.load_audio(file_path)
                
                if audio is not None:
                    features = self.extract_features(audio, sr)
                    mel_spec = features['mel_spectrogram']
                    
                    # Redimensionar a tamaño fijo
                    if mel_spec.shape[1] != 130:
                        # Ajustar dimensión temporal
                        if mel_spec.shape[1] < 130:
                            mel_spec = np.pad(mel_spec, ((0, 0), (0, 130 - mel_spec.shape[1])), mode='constant')
                        else:
                            mel_spec = mel_spec[:, :130]
                    
                    X.append(mel_spec)
                    y.append(category)
        
        if len(X) == 0:
            print("No hay datos suficientes para entrenar.")
            return None, None, None, None
        
        X = np.array(X)
        y = np.array(y)
        
        # Expandir dimensiones para CNN
        X = X[..., np.newaxis]
        
        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = keras.utils.to_categorical(y_encoded)
        
        # Dividir dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\nDataset preparado:")
        print(f"   - Total de muestras: {len(X)}")
        print(f"   - Entrenamiento: {X_train.shape[0]}")
        print(f"   - Validación: {X_test.shape[0]}")
        print(f"   - Forma de entrada: {X_train.shape[1:]}")
        
        return X_train, X_test, y_train, y_test
    
    def build_improved_model(self, input_shape, num_classes):
        """Construye un modelo CNN mejorado"""
        model = keras.Sequential([
            # Primera capa convolucional
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Segunda capa convolucional
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Tercera capa convolucional
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Capas densas
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Capa de salida
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compilar con learning rate ajustado
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def train_model(self, epochs=100, batch_size=16):
        """Entrena el modelo con el dataset"""
        print("\n🚀 Iniciando entrenamiento del modelo...")
        
        # Preparar datos
        X_train, X_test, y_train, y_test = self.prepare_training_data()
        
        if X_train is None:
            return {
                'success': False,
                'error': 'No hay suficientes datos para entrenar. Agrega al menos 5 audios por categoría.'
            }
        
        # Construir modelo
        self.model = self.build_improved_model(
            input_shape=X_train.shape[1:],
            num_classes=y_train.shape[1]
        )
        
        print("\n📐 Arquitectura del modelo:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(app.config['MODEL_FOLDER'], 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"\n🎯 Entrenando durante {epochs} épocas...")
        
        # Entrenar
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Guardar modelo
        self.save_model()
        
        print("\nEntrenamiento completado.")
        print(f"   - Accuracy: {results[1]*100:.2f}%")
        print(f"   - Precision: {results[2]*100:.2f}%")
        print(f"   - Recall: {results[3]*100:.2f}%")
        
        return {
            'success': True,
            'accuracy': float(results[1]),
            'precision': float(results[2]),
            'recall': float(results[3]),
            'loss': float(results[0]),
            'epochs_trained': len(history.history['loss'])
        }
    
    def save_model(self):
        """Guarda el modelo entrenado"""
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'hospital_model.h5')
        encoder_path = os.path.join(app.config['MODEL_FOLDER'], 'label_encoder.pkl')
        
        self.model.save(model_path)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Modelo guardado en: {model_path}")
    
    def predict_audio(self, file_path, threshold=0.90):
        """Predice la categoría del audio con umbral por clase y guardas"""
        print(f"\nAnalizando: {file_path}")
        
        # Cargar audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return {'error': 'No se pudo cargar el audio'}
        
        # Extraer características
        features = self.extract_features(audio, sr)
        mel_spec = features['mel_spectrogram']
        
        # Ajustar dimensiones
        if mel_spec.shape[1] != 130:
            if mel_spec.shape[1] < 130:
                mel_spec = np.pad(mel_spec, ((0, 0), (0, 130 - mel_spec.shape[1])), mode='constant')
            else:
                mel_spec = mel_spec[:, :130]
        
        X = mel_spec[np.newaxis, ..., np.newaxis]
        
        # Predecir
        if self.model is not None and self.label_encoder is not None:
            # Usar modelo entrenado
            predictions = self.model.predict(X, verbose=0)
        else:
            # Modo simulación con heurísticas mejoradas
            predictions = self._smart_simulation(features)
        
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_category = self.label_encoder.classes_[predicted_idx] if self.label_encoder else self.categories[predicted_idx]

        # Umbrales por clase (ajustados a nuevas categorías)
        # Umbrales por clase: usar el valor del slider uniformemente
        class_thresholds = {
            'tos': threshold,
            'grito_dolor': threshold,
            'alarma_medica': threshold,
        }

        # Determinar si es anomalía: cualquier evento detectado por encima del umbral
        is_anomaly = (predicted_category in ['tos', 'grito_dolor', 'alarma_medica']) and (confidence >= class_thresholds.get(predicted_category, threshold))
        # Etiqueta siempre basada en la categoría predicha
        predicted_label = self.category_names[predicted_category]
        
        # Preparar nombres de clases según el origen (modelo/encoder vs categorías locales)
        classes_order = list(self.label_encoder.classes_) if self.label_encoder is not None else list(self.categories)

        result = {
            'success': True,
            'predicted_label': predicted_label,
            'confidence': round(confidence * 100, 2),
            'is_anomaly': is_anomaly,
            'probabilities': {
                self.category_names.get(classes_order[i], classes_order[i]): round(float(predictions[0][i]) * 100, 2)
                for i in range(len(classes_order))
            },
            'features': {
                'zero_crossing_rate': round(features['zero_crossing_rate'], 4),
                'spectral_centroid': round(features['spectral_centroid'], 2),
                'spectral_rolloff': round(features['spectral_rolloff'], 2),
                'rms_energy': round(features['rms_energy'], 4),
                'tempo': round(features['tempo'], 2)
            },
            'timestamp': datetime.now().isoformat(),
            'model_trained': self.model is not None
        }
        
        print(f"Resultado: {predicted_label} ({confidence*100:.2f}%)")
        return result

    def predict_array(self, audio_array, sr, threshold=0.85):
        """Predice a partir de un arreglo PCM en memoria."""
        try:
            audio = np.asarray(audio_array, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            # Re-muestrear si es necesario
            target_sr = getattr(self, 'sample_rate', 22050)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

            # Extraer características y predecir reutilizando la lógica
            features = self.extract_features(audio, sr)
            mel_spec = features['mel_spectrogram']

            if mel_spec.shape[1] != 130:
                if mel_spec.shape[1] < 130:
                    mel_spec = np.pad(mel_spec, ((0, 0), (0, 130 - mel_spec.shape[1])), mode='constant')
                else:
                    mel_spec = mel_spec[:, :130]

            X = mel_spec[np.newaxis, ..., np.newaxis]

            if self.model is not None and self.label_encoder is not None:
                predictions = self.model.predict(X, verbose=0)
            else:
                predictions = self._smart_simulation(features)

            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            predicted_category = self.label_encoder.classes_[predicted_idx] if self.label_encoder else self.categories[predicted_idx]

            class_thresholds = {
                'tos': threshold,
                'grito_dolor': threshold,
                'alarma_medica': threshold,
            }

            is_anomaly = (predicted_category in ['tos', 'grito_dolor', 'alarma_medica']) and (confidence >= class_thresholds.get(predicted_category, threshold))
            predicted_label = self.category_names[predicted_category]

            classes_order = list(self.label_encoder.classes_) if self.label_encoder is not None else list(self.categories)

            result = {
                'success': True,
                'predicted_label': predicted_label,
                'confidence': round(confidence * 100, 2),
                'is_anomaly': is_anomaly,
                'probabilities': {
                    self.category_names.get(classes_order[i], classes_order[i]): round(float(predictions[0][i]) * 100, 2)
                    for i in range(len(classes_order))
                },
                'features': {
                    'zero_crossing_rate': round(features['zero_crossing_rate'], 4),
                    'spectral_centroid': round(features['spectral_centroid'], 2),
                    'spectral_rolloff': round(features['spectral_rolloff'], 2),
                    'rms_energy': round(features['rms_energy'], 4),
                    'tempo': round(features['tempo'], 2)
                },
                'timestamp': datetime.now().isoformat(),
                'model_trained': self.model is not None
            }

            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}
    def _smart_simulation(self, features):
        """Simulación inteligente basada en características acústicas (sin 'Normal' ni 'Caída')"""
        # Análisis heurístico de características
        zcr = features['zero_crossing_rate']
        energy = features['rms_energy']
        centroid = features['spectral_centroid']
        rolloff = features.get('spectral_rolloff', 0)
        bandwidth = features.get('spectral_bandwidth', 0)
        flatness = features.get('spectral_flatness', 0)

        # Inicializar probabilidades para ['tos', 'grito_dolor', 'alarma_medica']
        probs = np.zeros(len(self.categories))

        # Heurísticas más estrictas y balanceadas
        # Tos: ruido transitorio, zcr y flatness moderados, centroide medio
        if (zcr > 0.35 and energy > 0.12 and 1500 < centroid < 3500 and 0.45 < flatness < 0.65):
            probs[0] = 0.65
        # Grito de dolor: energía alta, centroid medio-alto, zcr moderado y flatness medio
        elif (energy > 0.25 and 2500 < centroid < 6000 and 0.15 < zcr < 0.45 and flatness > 0.35):
            probs[1] = 0.68
        # Alarma médica: tonal (flatness baja), alta frecuencia, rolloff alto, banda ancha
        elif (centroid > 6000 and zcr > 0.45 and rolloff > 9000 and energy > 0.10 and flatness < 0.35 and bandwidth > 1800):
            probs[2] = 0.72
        else:
            # Fallback neutral: distribución equilibrada
            probs[:] = np.array([0.34, 0.33, 0.33])

        # Normalizar y añadir ruido leve
        probs = probs + np.random.random(len(probs)) * 0.03
        probs = probs / probs.sum()

        return probs[np.newaxis, :]

# ============================================================================
# INSTANCIA GLOBAL DEL MONITOR
# ============================================================================

monitor = HospitalAudioMonitor()

# ============================================================================
# HTML TEMPLATE
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistema de Monitoreo Acústico Hospitalario</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --primary: #0066cc;
            --secondary: #00a651;
            --danger: #dc3545;
            --warning: #ffc107;
            --light: #f8f9fa;
            --dark: #212529;
            --shadow: rgba(0, 0, 0, 0.1);
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f4f6f8;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: white;
            padding: 36px;
            border-radius: 15px;
            box-shadow: 0 10px 30px var(--shadow);
            margin-bottom: 36px;
            display: flex;
            align-items: center;
            gap: 28px;
        }
        .logo {
            width: 100px; height: 100px;
            padding: 6px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 44px;
            letter-spacing: 1px;
            color: white;
        }
        .header-content h1 {
            color: var(--primary);
            font-size: 32px;
            margin-bottom: 8px;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }
        .tab {
            flex: 1;
            padding: 20px;
            background: white;
            border-radius: 12px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 5px 15px var(--shadow);
            font-weight: 600;
            color: #6c757d;
        }
        .tab:hover {
            transform: translateY(-3px);
        }
        .tab.active {
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px var(--shadow);
            margin-bottom: 30px;
        }
        .upload-zone {
            border: 3px dashed #dee2e6;
            border-radius: 12px;
            padding: 60px 40px;
            text-align: center;
            cursor: pointer;
            background: var(--light);
            transition: all 0.3s;
        }
        .upload-zone:hover {
            border-color: var(--primary);
            background: #e7f3ff;
            transform: scale(1.02);
        }
        input[type="file"], select { display: none; }
        .btn {
            padding: 15px 40px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 102, 204, 0.3);
        }
        .btn-secondary {
            background: var(--secondary);
            color: white;
        }
        .btn-secondary:hover {
            background: #008d42;
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .loading.active { display: block; }
        .spinner {
            width: 50px; height: 50px;
            border: 5px solid var(--light);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .results {
            display: none;
            animation: fadeIn 0.5s;
        }
        .results.active { display: block; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .result-card {
            padding: 30px;
            border-radius: 12px;
            color: white;
            margin-bottom: 20px;
        }
        .result-card.normal {
            background: linear-gradient(135deg, var(--secondary), #00c853);
        }
        .result-card.anomaly {
            background: linear-gradient(135deg, var(--danger), #ff4757);
        }
        @keyframes alertPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        .result-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        .result-icon {
            width: 60px; height: 60px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 30px;
        }
        .result-title { font-size: 28px; font-weight: bold; }
        .confidence-bar {
            background: rgba(255, 255, 255, 0.2);
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 15px;
        }
        .confidence-fill {
            height: 100%;
            background: white;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: var(--primary);
            transition: width 1s ease;
        }
        /* VU meter (nivel del micrófono) */
        .vu-container {
            width: 100%;
            height: 14px;
            background: rgba(0,0,0,0.08);
            border-radius: 8px;
            overflow: hidden;
        }
        .vu-fill {
            height: 100%;
            width: 0%;
            background: var(--secondary);
            transition: width 80ms linear, background 150ms ease;
        }
        .probabilities {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .probability-item {
            background: rgba(255, 255, 255, 0.15);
            padding: 15px;
            border-radius: 10px;
        }
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        .feature-box {
            background: var(--light);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .feature-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--primary);
            margin-bottom: 5px;
        }
        .alert {
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .alert-info {
            background: #d1ecf1;
            color: #0c5460;
            border-left: 4px solid var(--primary);
        }
        .alert-warning {
            background: #fff3cd;
            color: #856404;
            border-left: 4px solid var(--warning);
        }
        .alert-success {
            background: #d4edda;
            color: #155724;
            border-left: 4px solid var(--secondary);
        }
        .category-select {
            width: 100%;
            padding: 15px;
            border: 2px solid var(--border);
            border-radius: 8px;
            font-size: 16px;
            margin: 15px 0;
            background: white;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">SMAH</div>
            <div class="header-content">
                <h1>Sistema de Monitoreo Acústico Hospitalario</h1>
                <p>Detección inteligente de anomalías mediante Deep Learning | CNN + MFCC</p>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" data-tab="analyze">
                <div>Analizar Audio</div>
            </div>
            <div class="tab" data-tab="training">
                <div>Entrenar Modelo</div>
            </div>
            <div class="tab" data-tab="realtime">
                <div>Tiempo Real</div>
            </div>
        </div>
        
        <!-- Tab Analizar -->
        <div class="tab-content active" id="analyze">
            <div class="card">
                <div class="alert alert-info">
                    <span>Formatos permitidos: WAV, MP3, OGG, M4A, FLAC</span>
                </div>
                
                <div style="display:flex; align-items:center; gap:20px; margin:15px 0 25px;">
            <label for="thresholdInput" style="font-weight:600; color:#333;">Umbral de confianza:</label>
            <input type="range" id="thresholdInput" min="60" max="95" value="85"
                           style="flex:1;" />
            <span id="thresholdValue" style="min-width:60px; font-weight:600; color:#0066cc;">85%</span>
                </div>
                <div class="upload-zone" id="uploadZone">
                    <div style="font-size: 80px; margin-bottom: 20px;"></div>
                    <h2>Arrastra tu archivo aquí</h2>
                    <p style="color: #6c757d;">o haz clic para seleccionar</p>
                    <input type="file" id="fileInput" accept="audio/*">
                </div>
                
                <div style="text-align: center; margin-top: 20px;">
                    <button class="btn btn-primary" id="analyzeBtn" disabled>
                        <span>Analizar Audio</span>
                    </button>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <h3 style="color: white;">Analizando audio con Deep Learning...</h3>
            </div>
            
            <div class="results" id="results">
                <div class="card">
                    <div class="result-card" id="resultCard">
                        <div class="result-header">
                            <div class="result-icon" id="resultIcon">Normal</div>
                            <div>
                                <div class="result-title" id="resultTitle">—</div>
                                <div id="resultSubtitle" style="opacity: 0.9; margin-top: 5px;">Escuchando…</div>
                            </div>
                        </div>
                        
                        <div>
                            <strong>Nivel de Confianza</strong>
                            <div class="confidence-bar">
                                <div class="confidence-fill" id="confidenceFill">0%</div>
                            </div>
                        </div>
                        
                        <div class="probabilities" id="probabilities"></div>
                    </div>
                    
                    <h3 style="margin: 30px 0 20px;">Remedios sugeridos</h3>
                    <div class="features-grid" id="featuresGrid"></div>
                    
                    <div style="margin-top: 30px; text-align: center;">
                        <button class="btn btn-primary" onclick="location.reload()">
                            <span>Nuevo Análisis</span>
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Tab Entrenar -->
        <div class="tab-content" id="training">
            <div class="card">
                <h2 style="margin-bottom: 20px; color: var(--primary);">Gestión de Dataset</h2>
                
                <div class="alert alert-warning">
                    <span></span>
                    <span>Para entrenar el modelo necesitas al menos <strong>10-15 audios por categoría</strong>. Cuantos más, mejor será la precisión.</span>
                </div>
                
                <div style="margin-bottom: 30px;">
                    <h3 style="margin-bottom: 15px;">Agregar Audio al Dataset</h3>
                    <p style="color: #6c757d; margin-bottom: 15px;">Selecciona la categoría y sube el audio correspondiente:</p>
                    
                    <select class="category-select" id="categorySelect" style="display: block;">
                        <option value="">-- Selecciona una categoría --</option>
                        <option value="tos">Tos</option>
                        <option value="grito_dolor">Grito de Dolor</option>
                        <option value="alarma_medica">Alarma Médica</option>
                    </select>
                    
                    <div class="upload-zone" id="uploadZoneDataset" style="margin-top: 15px;">
                        <div style="font-size: 60px; margin-bottom: 15px;"></div>
                        <h3>Subir audio a dataset</h3>
                        <p style="color: #6c757d;">Arrastra o haz clic</p>
                        <input type="file" id="fileInputDataset" accept="audio/*">
                    </div>
                    <!-- Subida masiva por carpeta -->
                    <div class="upload-zone" id="uploadZoneDatasetFolder" style="margin-top: 15px;">
                        <div style="font-size: 60px; margin-bottom: 15px;"></div>
                        <h3>Subir carpeta de audios</h3>
                        <p style="color: #6c757d;">Selecciona una carpeta para subir múltiples audios</p>
                        <input type="file" id="folderInputDataset" webkitdirectory directory multiple accept="audio/*">
                    </div>
                </div>
                
                <div id="datasetStats" style="margin-bottom: 30px;">
                    <h3 style="margin-bottom: 15px;">Estado del Dataset</h3>
                    <div class="features-grid" id="statsGrid">
                        <div class="feature-box">
                            <div class="feature-value" id="stat-tos">0</div>
                            <div class="feature-label">Tos</div>
                        </div>
                        <div class="feature-box">
                            <div class="feature-value" id="stat-grito_dolor">0</div>
                            <div class="feature-label">Grito de Dolor</div>
                        </div>
                        <div class="feature-box">
                            <div class="feature-value" id="stat-alarma_medica">0</div>
                            <div class="feature-label">Alarma Médica</div>
                        </div>
                    </div>
                </div>
                
                <div style="text-align: center;">
                    <button class="btn btn-secondary" id="trainBtn" style="margin-right: 10px;">
                        <span>Entrenar Modelo</span>
                    </button>
                    <button class="btn btn-secondary" id="seedDatasetBtn" style="margin-right: 10px;">
                        <span>Cargar dataset de ejemplo</span>
                    </button>
                    <button class="btn btn-primary" id="refreshStatsBtn">
                        <span>Actualizar Estadísticas</span>
                    </button>
                </div>

                <div id="seedStatus" style="margin-top: 12px; text-align: center; color: #6c757d;"></div>
                
                <div id="trainingResults" style="margin-top: 30px; display: none;">
                    <div class="alert alert-success">
                        <span></span>
                        <span id="trainingMessage">Modelo entrenado exitosamente</span>
                    </div>
                    <div class="features-grid" id="trainingMetrics"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab Tiempo Real -->
        <div class="tab-content" id="realtime">
            <div class="card">
                <h2 style="margin-bottom: 15px; color: var(--primary);">Monitoreo en Tiempo Real</h2>
                <div class="alert alert-info" style="margin-bottom:15px;">
                    <span></span>
                    <span>Permite acceso al micrófono para analizar en vivo. Usa auriculares para evitar retroalimentación.</span>
                </div>
                <div style="display:flex; align-items:center; gap:20px; margin:15px 0 25px;">
            <label for="thresholdInputRealtime" style="font-weight:600; color:#333;">Umbral de confianza:</label>
            <input type="range" id="thresholdInputRealtime" min="60" max="95" value="85" style="flex:1;" />
            <span id="thresholdValueRealtime" style="min-width:60px; font-weight:600; color:#0066cc;">85%</span>
                </div>

                <div style="display:flex; align-items:center; gap:15px; margin-bottom:20px;">
                    <button class="btn btn-primary" id="startRealtimeBtn">
                        <span>Iniciar Monitoreo</span>
                    </button>
                    <button class="btn btn-secondary" id="stopRealtimeBtn" disabled>
                        <span>Detener</span>
                    </button>
                    <div id="realtimeStatus" style="margin-left:auto; font-weight:600; color:#6c757d;">Estado: Inactivo</div>
                </div>

                <!-- Indicador de nivel del micrófono (VU meter) -->
                <div style="margin: 0 0 20px;">
                    <strong>Nivel del micrófono</strong>
                    <div class="vu-container" id="vuContainer">
                        <div class="vu-fill" id="vuFill"></div>
                    </div>
                    <div id="vuLabel" style="margin-top:6px; font-size:14px; color:#6c757d;">Silencio</div>
                </div>

                <div class="results active" id="realtimeResults">
                    <div class="card" style="padding:20px;">
                        <div class="result-header">
                            <div class="result-icon" id="rtIcon">Normal</div>
                            <div>
                                <div class="result-title" id="rtTitle">En espera…</div>
                                <div id="rtSubtitle" style="opacity:0.9; margin-top:5px;">Muestreando audio…</div>
                            </div>
                        </div>
                        <div>
                            <strong>Nivel de Confianza</strong>
                            <div class="confidence-bar">
                                <div class="confidence-fill" id="rtConfidence">0%</div>
                            </div>
                        </div>
                        <div class="probabilities" id="rtProbabilities"></div>
                        <h3 style="margin: 20px 0 10px;">Remedios sugeridos</h3>
                        <div class="features-grid" id="rtFeatures"></div>
                    </div>
                </div>
            </div>
        </div>
    
    <script>
        let selectedFile = null;
        let selectedCategory = null;
        
        // Tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab).classList.add('active');
                if (tab.dataset.tab === 'training') {
                    updateDatasetStats();
                }
            });
        });
        
        // Upload zona analizar
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        uploadZone.addEventListener('click', () => fileInput.click());
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '#0066cc';
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.style.borderColor = '#dee2e6';
        });
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '#dee2e6';
            if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) handleFile(e.target.files[0]);
        });
        
        function handleFile(file) {
            selectedFile = file;
            uploadZone.innerHTML = `
            <div style="font-size: 60px; margin-bottom: 15px;"></div>
                <h2>Archivo seleccionado</h2>
                <p style="color: #6c757d;"><strong>${file.name}</strong></p>
                <p style="color: #6c757d;">${(file.size / 1024 / 1024).toFixed(2)} MB</p>
            `;
            analyzeBtn.disabled = false;
        }
        
        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            document.getElementById('loading').classList.add('active');
            analyzeBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('audio', selectedFile);
            const th = parseFloat(document.getElementById('thresholdInput').value) / 100.0;
            formData.append('threshold', th.toString());
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data);
                } else {
                    alert('Error: ' + (data.error || 'Error desconocido'));
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading').classList.remove('active');
            }
        });
        
        function displayResults(data) {
            const resultCard = document.getElementById('resultCard');
            const resultIcon = document.getElementById('resultIcon');
            const resultTitle = document.getElementById('resultTitle');
            const resultSubtitle = document.getElementById('resultSubtitle');
            const confidenceFill = document.getElementById('confidenceFill');
            const probabilities = document.getElementById('probabilities');
            const featuresGrid = document.getElementById('featuresGrid');
            
            resultCard.className = 'result-card ' + (data.is_anomaly ? 'anomaly' : 'normal');
            resultIcon.textContent = data.is_anomaly ? 'Alerta' : 'Normal';
            resultTitle.textContent = data.predicted_label;
            const infoLine = `Archivo: ${data.filename || '—'} · Confianza: ${data.confidence}%`;
            resultSubtitle.textContent = (data.is_anomaly ? 
                'ANOMALÍA DETECTADA - Notificar al personal médico' : 
                'Situación normal - Sin alertas') + ` \u2013 ${infoLine}`;
            
            if (!data.model_trained) {
                resultSubtitle.textContent += ' (Modo simulación - Entrena el modelo para mejor precisión)';
            }
            
            confidenceFill.style.width = data.confidence + '%';
            confidenceFill.textContent = data.confidence + '%';
            
            probabilities.innerHTML = '';
            Object.entries(data.probabilities).forEach(([cat, prob]) => {
                probabilities.innerHTML += `
                    <div class="probability-item">
                        <div style="font-size: 14px; margin-bottom: 8px; opacity: 0.9;">${cat}</div>
                        <div style="font-size: 22px; font-weight: bold;">${prob}%</div>
                    </div>
                `;
            });
            
            const remedies = getRemediesForLabel(data.predicted_label);
            featuresGrid.innerHTML = remedies.map(r => `
                <div class="feature-box">
                    <div class="feature-value">${r}</div>
                    <div class="feature-label">Recomendación</div>
                </div>
            `).join('');
            
            document.getElementById('results').classList.add('active');
        }
        
        function getCategoryIcon(category) {
            return '';
        }
        
        function getRemediesForLabel(label) {
            const remediesMap = {
                'Tos': [
                    'Beber líquidos tibios con frecuencia',
                    'Pastillas/laringeas para la garganta',
                    'Inhalación de vapor (ducha caliente)',
                    'Evitar humo y polvo; ventilar el ambiente',
                    'Consultar si dura >48h, fiebre alta o dificultad respiratoria'
                ],
                'Grito de Dolor': [
                    'Evaluar escala de dolor (0–10)',
                    'Avisar a enfermería/médico de turno',
                    'Administrar analgésicos según protocolo',
                    'Aplicar frío/calor local si está indicado',
                    'Revisar herida o zona afectada y signos de alarma'
                ],
                'Alarma Médica': [
                    'Notificar inmediatamente al personal',
                    'Verificar equipos y pacientes cercanos',
                    'Seguir el protocolo de emergencia del servicio'
                ]
            };
            return remediesMap[label] || ['Sin síntomas claros', 'Continuar monitorización'];
        }
        
        // Dataset management
        // Control visual de umbral
        const thresholdInput = document.getElementById('thresholdInput');
        const thresholdValue = document.getElementById('thresholdValue');
        thresholdInput.addEventListener('input', () => {
            thresholdValue.textContent = `${thresholdInput.value}%`;
        });

        // Tiempo Real: controles de umbral
        const thresholdInputRealtime = document.getElementById('thresholdInputRealtime');
        const thresholdValueRealtime = document.getElementById('thresholdValueRealtime');
        if (thresholdInputRealtime && thresholdValueRealtime) {
            thresholdInputRealtime.addEventListener('input', () => {
                thresholdValueRealtime.textContent = `${thresholdInputRealtime.value}%`;
            });
        }
        const categorySelect = document.getElementById('categorySelect');
        const uploadZoneDataset = document.getElementById('uploadZoneDataset');
        const fileInputDataset = document.getElementById('fileInputDataset');
        const uploadZoneDatasetFolder = document.getElementById('uploadZoneDatasetFolder');
        const folderInputDataset = document.getElementById('folderInputDataset');
        
        categorySelect.addEventListener('change', () => {
            selectedCategory = categorySelect.value;
        });
        
        uploadZoneDataset.addEventListener('click', () => {
            if (!selectedCategory) {
                alert('Por favor selecciona una categoría primero');
                return;
            }
            fileInputDataset.click();
        });

        // Click para seleccionar carpeta completa
        uploadZoneDatasetFolder.addEventListener('click', () => {
            if (!selectedCategory) {
                alert('Por favor selecciona una categoría primero');
                return;
            }
            folderInputDataset.click();
        });
        
        uploadZoneDataset.addEventListener('dragover', (e) => {
            e.preventDefault();
            if (selectedCategory) {
                uploadZoneDataset.style.borderColor = '#0066cc';
            }
        });
        
        uploadZoneDataset.addEventListener('dragleave', () => {
            uploadZoneDataset.style.borderColor = '#dee2e6';
        });
        
        uploadZoneDataset.addEventListener('drop', async (e) => {
            e.preventDefault();
            uploadZoneDataset.style.borderColor = '#dee2e6';
            if (!selectedCategory) {
                alert('Selecciona una categoría primero');
                return;
            }
            if (e.dataTransfer.files[0]) {
                await uploadToDataset(e.dataTransfer.files[0]);
            }
        });
        
        fileInputDataset.addEventListener('change', async (e) => {
            if (e.target.files[0] && selectedCategory) {
                await uploadToDataset(e.target.files[0]);
            }
        });

        // Selección de carpeta: envía todos los audios de la carpeta
        folderInputDataset.addEventListener('change', async (e) => {
            const files = Array.from(e.target.files || []);
            if (!selectedCategory) {
                alert('Selecciona una categoría primero');
                return;
            }
            const audioFiles = files.filter(f => f.type.startsWith('audio') || /\.(wav|mp3|ogg|m4a|flac)$/i.test(f.name));
            if (audioFiles.length === 0) {
                alert('No se encontraron archivos de audio en la carpeta seleccionada');
                return;
            }
            await uploadBulkToDataset(audioFiles);
            folderInputDataset.value = '';
        });
        
        async function uploadToDataset(file) {
            const formData = new FormData();
            formData.append('audio', file);
            formData.append('category', selectedCategory);
            
            try {
                const response = await fetch('/api/dataset/add', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
            alert(`Audio agregado a la categoría: ${data.category_name}`);
                    updateDatasetStats();
                    fileInputDataset.value = '';
                } else {
                    alert('Error: ' + (data.error || 'Error desconocido'));
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        async function uploadBulkToDataset(files) {
            const formData = new FormData();
            formData.append('category', selectedCategory);
            files.forEach(f => formData.append('audios', f, f.name));
            try {
                const response = await fetch('/api/dataset/add_bulk', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.success) {
                    const saved = data.saved || 0;
            alert(`${saved} audios agregados a la categoría: ${data.category_name}`);
                    updateDatasetStats();
                } else {
                    alert('Error: ' + (data.error || 'Error desconocido'));
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function updateDatasetStats() {
            try {
                const response = await fetch('/api/dataset/stats');
                const data = await response.json();
                
                if (data.success) {
                    Object.entries(data.stats).forEach(([cat, count]) => {
                        const elem = document.getElementById(`stat-${cat}`);
                        if (elem) elem.textContent = count;
                    });
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }
        
        document.getElementById('refreshStatsBtn').addEventListener('click', updateDatasetStats);

        // Siembra de dataset de ejemplo
        const seedBtn = document.getElementById('seedDatasetBtn');
        const seedStatus = document.getElementById('seedStatus');
        if (seedBtn) {
            seedBtn.addEventListener('click', async () => {
                try {
                    seedBtn.disabled = true;
            seedBtn.innerHTML = '<span>Sembrando…</span>';
                    seedStatus.textContent = 'Sembrando dataset de ejemplo…';
                    const resp = await fetch('/api/dataset/seed', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ per_class: 8 })
                    });
                    const data = await resp.json();
                    if (data.status === 'ok') {
                        const total = Object.values(data.counts || {}).reduce((a,b)=>a+b,0);
            seedStatus.textContent = `Listo: ${total} audios generados en ${data.dataset_dir}`;
                        updateDatasetStats();
                    } else {
            seedStatus.textContent = 'Error al sembrar dataset.';
                    }
                } catch (e) {
                    console.error(e);
            seedStatus.textContent = 'Error al sembrar dataset.';
                } finally {
                    seedBtn.disabled = false;
            seedBtn.innerHTML = '<span>Cargar dataset de ejemplo</span>';
                }
            });
        }
        
        document.getElementById('trainBtn').addEventListener('click', async () => {
            if (!confirm('¿Estás seguro de iniciar el entrenamiento? Esto puede tardar varios minutos.')) {
                return;
            }
            
            const trainBtn = document.getElementById('trainBtn');
            trainBtn.disabled = true;
            trainBtn.innerHTML = '<span>Entrenando...</span>';
            
            try {
                const response = await fetch('/api/train', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const resultsDiv = document.getElementById('trainingResults');
                    const metricsDiv = document.getElementById('trainingMetrics');
                    const messageSpan = document.getElementById('trainingMessage');
                    
                    messageSpan.textContent = `Modelo entrenado exitosamente en ${data.epochs_trained} épocas`;
                    
                    metricsDiv.innerHTML = `
                        <div class="feature-box">
                            <div class="feature-value">${(data.accuracy * 100).toFixed(2)}%</div>
                            <div class="feature-label">Accuracy</div>
                        </div>
                        <div class="feature-box">
                            <div class="feature-value">${(data.precision * 100).toFixed(2)}%</div>
                            <div class="feature-label">Precision</div>
                        </div>
                        <div class="feature-box">
                            <div class="feature-value">${(data.recall * 100).toFixed(2)}%</div>
                            <div class="feature-label">Recall</div>
                        </div>
                        <div class="feature-box">
                            <div class="feature-value">${data.loss.toFixed(4)}</div>
                            <div class="feature-label">Loss</div>
                        </div>
                    `;
                    
                    resultsDiv.style.display = 'block';
                    
                    setTimeout(() => {
                        alert('Modelo entrenado. Ahora puedes analizar audios con mayor precisión.');
                    }, 500);
                } else {
                    alert('Error: ' + (data.error || 'Error desconocido'));
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                trainBtn.disabled = false;
                trainBtn.innerHTML = 'Entrenar Modelo';
            }
        });
        
        // Cargar stats al inicio
        updateDatasetStats();

        // ==========================
        // Tiempo Real: captura y envío
        // ==========================
        let audioContext = null;
        let mediaStream = null;
        let processor = null;
        let isStreaming = false;
        let sampleBuffer = [];
        let analyser = null;
        let vuRAF = null;
        const vuFill = document.getElementById('vuFill');
        const vuLabel = document.getElementById('vuLabel');
        const startBtn = document.getElementById('startRealtimeBtn');
        const stopBtn = document.getElementById('stopRealtimeBtn');
        const statusLabel = document.getElementById('realtimeStatus');

        async function startRealtime() {
            if (isStreaming) return;
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const source = audioContext.createMediaStreamSource(mediaStream);
                // Nodo analizador para VU meter
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 1024;
                analyser.smoothingTimeConstant = 0.8;
                source.connect(analyser);
                // Tamaño de buffer; 4096 proporciona bloques razonables
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                source.connect(processor);
                processor.connect(audioContext.destination);
                const sr = audioContext.sampleRate;
                const windowMs = 1000; // enviar cada ~1s
                const windowSamples = Math.floor(sr * windowMs / 1000);
                sampleBuffer = [];
                processor.onaudioprocess = (e) => {
                    const input = e.inputBuffer.getChannelData(0);
                    // Copiar a un array normal para serializar
                    for (let i = 0; i < input.length; i++) sampleBuffer.push(input[i]);
                    if (sampleBuffer.length >= windowSamples) {
                        const batch = sampleBuffer.slice(0, windowSamples);
                        sampleBuffer = sampleBuffer.slice(windowSamples);
            const th = thresholdInputRealtime ? (parseFloat(thresholdInputRealtime.value) / 100.0) : 0.85;
                        sendRealtimeBatch(batch, sr, th);
                    }
                };
                isStreaming = true;
                startBtn.disabled = true;
                stopBtn.disabled = false;
                statusLabel.textContent = 'Estado: Monitoreando';
                statusLabel.style.color = '#00a651';
                // Iniciar VU meter
                startVUMeter();
                // UI inicial
                updateRealtimeUI({
                    is_anomaly: false,
                    predicted_label: 'Escuchando…',
                    confidence: 0,
                    probabilities: {},
                    features: { zero_crossing_rate: 0, spectral_centroid: 0, spectral_rolloff: 0, rms_energy: 0, tempo: 0 },
                    model_trained: true,
                    success: true
                });
            } catch (err) {
                alert('No se pudo acceder al micrófono: ' + err.message);
            }
        }

        function stopRealtime() {
            if (!isStreaming) return;
            try {
                processor && processor.disconnect();
                if (mediaStream) {
                    mediaStream.getTracks().forEach(t => t.stop());
                }
                audioContext && audioContext.close();
            } finally {
                isStreaming = false;
                startBtn.disabled = false;
                stopBtn.disabled = true;
                statusLabel.textContent = 'Estado: Inactivo';
                statusLabel.style.color = '#6c757d';
                stopVUMeter();
                if (vuFill) vuFill.style.width = '0%';
                if (vuLabel) vuLabel.textContent = 'Inactivo';
            }
        }

        function startVUMeter() {
            if (!analyser || !vuFill) return;
            const buffer = new Float32Array(analyser.fftSize);
            const update = () => {
                if (!analyser) return;
                analyser.getFloatTimeDomainData(buffer);
                let sum = 0;
                for (let i = 0; i < buffer.length; i++) {
                    const v = buffer[i];
                    sum += v * v;
                }
                const rms = Math.sqrt(sum / buffer.length);
                const level = Math.min(100, Math.max(0, Math.round(rms * 200)));
                vuFill.style.width = level + '%';
                vuFill.style.background = level > 70 ? '#dc3545' : (level > 40 ? '#ffc107' : '#00a651');
                if (vuLabel) vuLabel.textContent = level > 10 ? 'Detectando sonido' : 'Silencio';
                vuRAF = requestAnimationFrame(update);
            };
            update();
        }

        function stopVUMeter() {
            if (vuRAF) cancelAnimationFrame(vuRAF);
            vuRAF = null;
        }

        async function sendRealtimeBatch(batch, sr, threshold) {
            try {
                const response = await fetch('/api/analyze_stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ samples: batch, sr: sr, threshold: threshold })
                });
                const data = await response.json();
                if (data && data.success) {
                    updateRealtimeUI(data);
                }
            } catch (error) {
                console.error('Error enviando lote tiempo real:', error);
            }
        }

        function updateRealtimeUI(data) {
            const icon = document.getElementById('rtIcon');
            const title = document.getElementById('rtTitle');
            const subtitle = document.getElementById('rtSubtitle');
            const conf = document.getElementById('rtConfidence');
            const probs = document.getElementById('rtProbabilities');
            const feats = document.getElementById('rtFeatures');
            icon.textContent = data.is_anomaly ? 'Alerta' : 'Normal';
            title.textContent = data.predicted_label || '—';
            const subInfo = `Predicción: ${data.predicted_label || '—'} (${(data.confidence || 0)}%)`;
            subtitle.textContent = (data.is_anomaly ? 'ANOMALÍA EN TIEMPO REAL' : 'Escuchando…') + ` \u2013 ${subInfo}`;
            conf.style.width = (data.confidence || 0) + '%';
            conf.textContent = (data.confidence || 0) + '%';
            probs.innerHTML = '';
            Object.entries(data.probabilities || {}).forEach(([cat, prob]) => {
                probs.innerHTML += `<div class="probability-item"><div style="font-size:14px; margin-bottom:8px; opacity:0.9;">${cat}</div><div style="font-size:22px; font-weight:bold;">${prob}%</div></div>`;
            });
            const rtRemedies = getRemediesForLabel(data.predicted_label || '');
            feats.innerHTML = rtRemedies.map(r => `
                <div class="feature-box"><div class="feature-value">${r}</div><div class="feature-label">Recomendación</div></div>
            `).join('');
        }

        if (startBtn && stopBtn) {
            startBtn.addEventListener('click', startRealtime);
            stopBtn.addEventListener('click', stopRealtime);
        }
    </script>
</body>
</html>
"""

# ============================================================================
# RUTAS DE FLASK
# ============================================================================

@app.route('/')
def index():
    """Página principal"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """Endpoint para analizar audio"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No se envió ningún archivo'})
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Nombre de archivo vacío'})
        
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if ext not in app.config['ALLOWED_EXTENSIONS']:
            return jsonify({'success': False, 'error': f'Formato no permitido'})
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{timestamp}_{filename}')
        file.save(filepath)
        
        # Umbral opcional desde el formulario (0.0 - 1.0)
        try:
            threshold_str = request.form.get('threshold', '0.85')
            threshold_val = float(threshold_str)
        except Exception:
            threshold_val = 0.85

        result = monitor.predict_audio(filepath, threshold=threshold_val)
        # Incluir el nombre original del archivo analizado para claridad en UI
        result['filename'] = filename

        return jsonify(result)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analyze_stream', methods=['POST'])
def analyze_stream():
    """Analiza audio en tiempo real desde muestras PCM (JSON)."""
    try:
        data = request.get_json(force=True)
        samples = data.get('samples')
        sr = int(data.get('sr', 44100))
        threshold_val = float(data.get('threshold', 0.85))
        if samples is None:
            return jsonify({'success': False, 'error': 'No se enviaron muestras'})

        result = monitor.predict_array(samples, sr=sr, threshold=threshold_val)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dataset/add_bulk', methods=['POST'])
def add_bulk_to_dataset():
    """Agrega múltiples audios al dataset desde una sola carpeta"""
    try:
        if 'category' not in request.form:
            return jsonify({'success': False, 'error': 'Falta la categoría'})
        category = request.form['category']
        if category not in CATEGORIES:
            return jsonify({'success': False, 'error': 'Categoría inválida'})

        files = request.files.getlist('audios')
        if not files:
            return jsonify({'success': False, 'error': 'No se enviaron archivos'})

        saved = 0
        filenames = []
        for file in files:
            if not file:
                continue
            filename = secure_filename(file.filename)
            # Validar extensión
            ext = os.path.splitext(filename)[1].lower().lstrip('.')
            if ext not in app.config['ALLOWED_EXTENSIONS']:
                continue
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(app.config['DATASET_FOLDER'], category, f'{timestamp}_{filename}')
            file.save(filepath)
            saved += 1
            filenames.append(filename)

        return jsonify({
            'success': True,
            'category': category,
            'category_name': monitor.category_names[category],
            'saved': saved,
            'filenames': filenames
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dataset/add', methods=['POST'])
def add_to_dataset():
    """Agrega un audio al dataset"""
    try:
        if 'audio' not in request.files or 'category' not in request.form:
            return jsonify({'success': False, 'error': 'Faltan parámetros'})
        
        file = request.files['audio']
        category = request.form['category']
        
        if category not in CATEGORIES:
            return jsonify({'success': False, 'error': 'Categoría inválida'})
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(app.config['DATASET_FOLDER'], category, f'{timestamp}_{filename}')
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'category': category,
            'category_name': monitor.category_names[category],
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dataset/stats', methods=['GET'])
def get_dataset_stats():
    """Obtiene estadísticas del dataset"""
    try:
        stats = {}
        for cat in CATEGORIES:
            cat_path = os.path.join(app.config['DATASET_FOLDER'], cat)
            files = [f for f in os.listdir(cat_path) if f.endswith(tuple(['.'+ext for ext in app.config['ALLOWED_EXTENSIONS']]))]
            stats[cat] = len(files)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'total': sum(stats.values())
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dataset/seed', methods=['POST'])
def seed_dataset():
    """Genera audios sintéticos por categoría y los guarda en el dataset."""
    try:
        import numpy as np
        import wave
        # Parámetros
        if request.is_json:
            per_class = int(request.json.get('per_class', 8))
        else:
            per_class = int(request.form.get('per_class', 8))
        duration_sec = 3.0
        sample_rate = 16000
        n = int(duration_sec * sample_rate)

        base_dir = app.config['DATASET_FOLDER']
        classes = list(CATEGORIES)

        def write_wav(filepath, samples):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            clipped = np.clip(samples, -1.0, 1.0)
            int16 = (clipped * 32767).astype(np.int16)
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(int16.tobytes())

        def gen_noise(amp):
            return (np.random.randn(n).astype(np.float32)) * amp

        def gen_sine(freq, amp):
            t = np.arange(n) / sample_rate
            return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

        # Categorías eliminadas: normal y caída

        def gen_tos():
            x = np.zeros(n, dtype=np.float32)
            bursts = np.random.choice([2, 3])
            pos = 0
            for _ in range(bursts):
                b_len = int(np.random.uniform(0.12, 0.25) * sample_rate)
                gap = int(np.random.uniform(0.08, 0.18) * sample_rate)
                pos = min(pos + gap, n - b_len)
                burst = gen_noise(0.35)[:b_len]
                tt = np.linspace(0, 1, b_len)
                env = np.minimum(1.0, (tt * 12)) * np.exp(-3 * tt)
                x[pos:pos + b_len] += (burst * env.astype(np.float32))
                pos += b_len
            return x

        def gen_alarma():
            x = np.zeros(n, dtype=np.float32)
            cycle_on = int(0.25 * sample_rate)
            cycle_off = int(0.25 * sample_rate)
            i = 0
            while i < n:
                end_on = min(i + cycle_on, n)
                x[i:end_on] += gen_sine(1000, 0.4)[:end_on - i]
                i = end_on + cycle_off
            return x

        def gen_grito():
            base = gen_noise(0.28)
            hp = np.zeros(n, dtype=np.float32)
            hp[1:] = base[1:] - 0.95 * base[:-1]
            tt = np.linspace(0, 1, n)
            env = (0.6 + 0.4 * (1 - np.exp(-4 * tt))).astype(np.float32)
            return hp * env

        gen_map = {
            'tos': gen_tos,
            'alarma_medica': gen_alarma,
            'grito_dolor': gen_grito,
        }

        counts = {c: 0 for c in classes}
        for cls in classes:
            cls_dir = os.path.join(base_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(per_class):
                samples = gen_map[cls]()
                gain = float(np.random.uniform(0.9, 1.1))
                samples = (samples * gain).astype(np.float32)
                fname = f"{cls}_{i+1:02d}.wav"
                fpath = os.path.join(cls_dir, fname)
                write_wav(fpath, samples)
                counts[cls] += 1

        return jsonify({"status": "ok", "dataset_dir": base_dir, "counts": counts})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route('/api/train', methods=['POST'])
def train_model():
    """Entrena el modelo"""
    try:
        result = monitor.train_model(epochs=100, batch_size=16)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal"""
    print("\n" + "="*70)
    print("SISTEMA DE MONITOREO ACÚSTICO HOSPITALARIO")
    print("="*70)
    print("\nSistema iniciado")
    print(f"Categorías: {', '.join([monitor.category_names[c] for c in CATEGORIES])}")
    print(f"\nCarpetas:")
    print(f"   - Dataset: {app.config['DATASET_FOLDER']}")
    print(f"   - Modelos: {app.config['MODEL_FOLDER']}")
    print(f"   - Uploads: {app.config['UPLOAD_FOLDER']}")
    print("\nINSTRUCCIONES:")
    print("   1. Ve a la pestaña 'Entrenar Modelo'")
    print("   2. Sube al menos 10-15 audios por categoría")
    print("   3. Haz clic en 'Entrenar Modelo'")
    print("   4. Una vez entrenado, analiza nuevos audios con alta precisión")
    print("\nServidor Flask iniciando...")
    print("Abre tu navegador en: http://localhost:5000")
    print("\nPresiona Ctrl+C para detener\n")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()