import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import cv2
import io
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache_lock = threading.Lock()
        self._kmeans_cache = {}
        self._mean_shift_cache = {}
        self._dbscan_cache = {}

    def __del__(self):
        self.executor.shutdown()

    def load_audio(self, file_path, sr=22050):
        """Загрузка аудиофайла"""
        try:
            y, sr = librosa.load(file_path, sr=sr)
            return y, sr
        except Exception as e:
            logger.error(f"Ошибка при загрузке аудио: {str(e)}")
            raise

    def extract_features(self, y, sr):
        """Извлечение признаков из аудио для кластеризации"""
        try:
            # Извлекаем различные признаки
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            
            # Объединяем все признаки
            features = np.vstack([mfcc, chroma, spectral_contrast, tonnetz])
            
            # Транспонируем для получения формы (время, признаки)
            features = features.T
            
            return features
        except Exception as e:
            logger.error(f"Ошибка при извлечении признаков: {str(e)}")
            raise

    def apply_kmeans(self, y, sr, k=5, attempts=10):
        """K-means кластеризация аудио сигнала"""
        try:
            cache_key = (hash(y.tobytes()), sr, k, attempts)
            with self._cache_lock:
                if cache_key in self._kmeans_cache:
                    return self._kmeans_cache[cache_key]

            # Извлекаем признаки
            features = self.extract_features(y, sr)
            
            # Нормализуем признаки
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Применяем K-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(
                features_scaled.astype(np.float32), 
                k, 
                None, 
                criteria, 
                attempts, 
                cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Создаем спектрограмму для визуализации
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            
            # Создаем цветную карту кластеров
            time_frames = D.shape[1]
            cluster_map = np.zeros((D.shape[0], time_frames, 3), dtype=np.uint8)
            
            # Интерполируем метки кластеров на временные рамки спектрограммы
            if len(labels) != time_frames:
                # Ресэмплируем метки для соответствия времени
                x_old = np.linspace(0, 1, len(labels))
                x_new = np.linspace(0, 1, time_frames)
                labels_resampled = np.interp(x_new, x_old, labels.flatten())
                labels_resampled = labels_resampled.astype(int)
            else:
                labels_resampled = labels.flatten()
            
            # Назначаем цвета кластерам
            colors = np.random.randint(0, 255, (k, 3), dtype=np.uint8)
            for i in range(time_frames):
                cluster_id = labels_resampled[i] % k
                cluster_map[:, i] = colors[cluster_id]
            
            result = {
                'labels': labels,
                'centers': centers,
                'cluster_map': cluster_map,
                'features': features_scaled,
                'spectrogram': D
            }
            
            with self._cache_lock:
                self._kmeans_cache[cache_key] = result
            
            logger.info(f"K-means кластеризация выполнена успешно с {k} кластерами")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при применении K-means: {str(e)}", exc_info=True)
            raise

    def apply_mean_shift(self, y, sr, bandwidth=0.5, max_iterations=300):
        """Mean Shift кластеризация аудио сигнала"""
        try:
            cache_key = (hash(y.tobytes()), sr, bandwidth, max_iterations)
            with self._cache_lock:
                if cache_key in self._mean_shift_cache:
                    return self._mean_shift_cache[cache_key]

            # Извлекаем признаки
            features = self.extract_features(y, sr)
            
            # Нормализуем признаки
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Применяем Mean Shift (используем имитацию через K-means с автоматическим определением k)
            # Поскольку в OpenCV нет прямого Mean Shift для произвольных данных,
            # мы используем адаптированный подход
            
            # Определяем оптимальное количество кластеров
            max_k = min(20, len(features_scaled) // 10)
            best_k = self._estimate_optimal_clusters(features_scaled, max_k)
            
            # Применяем кластеризацию
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, 0.1)
            _, labels, centers = cv2.kmeans(
                features_scaled.astype(np.float32), 
                best_k, 
                None, 
                criteria, 
                10, 
                cv2.KMEANS_PP_CENTERS
            )
            
            # Создаем спектрограмму для визуализации
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            
            # Создаем цветную карту кластеров
            time_frames = D.shape[1]
            cluster_map = np.zeros((D.shape[0], time_frames, 3), dtype=np.uint8)
            
            # Интерполируем метки кластеров
            if len(labels) != time_frames:
                x_old = np.linspace(0, 1, len(labels))
                x_new = np.linspace(0, 1, time_frames)
                labels_resampled = np.interp(x_new, x_old, labels.flatten())
                labels_resampled = labels_resampled.astype(int)
            else:
                labels_resampled = labels.flatten()
            
            # Назначаем цвета кластерам с использованием colormap
            colors = cv2.applyColorMap(
                np.linspace(0, 255, best_k).astype(np.uint8).reshape(-1, 1, 1), 
                cv2.COLORMAP_JET
            ).reshape(-1, 3)
            
            for i in range(time_frames):
                cluster_id = labels_resampled[i] % best_k
                cluster_map[:, i] = colors[cluster_id]
            
            result = {
                'labels': labels,
                'centers': centers,
                'cluster_map': cluster_map,
                'features': features_scaled,
                'spectrogram': D,
                'n_clusters': best_k
            }
            
            with self._cache_lock:
                self._mean_shift_cache[cache_key] = result
            
            logger.info(f"Mean Shift кластеризация выполнена успешно с {best_k} кластерами")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при применении Mean Shift: {str(e)}", exc_info=True)
            raise

    def apply_dbscan(self, y, sr, eps=0.5, min_samples=10):
        """DBSCAN кластеризация аудио сигнала"""
        try:
            cache_key = (hash(y.tobytes()), sr, eps, min_samples)
            with self._cache_lock:
                if cache_key in self._dbscan_cache:
                    return self._cache_lock[cache_key]

            # Извлекаем признаки
            features = self.extract_features(y, sr)
            
            # Нормализуем признаки
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Применяем сэмплирование для больших данных (аналогично image processor)
            sample_size = 10000 if len(features_scaled) > 10000 else None
            
            # Применяем DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
            
            if sample_size:
                # Используем сэмплирование
                sample_indices = np.random.choice(len(features_scaled), sample_size, replace=False)
                sample_features = features_scaled[sample_indices]
                sample_labels = dbscan.fit_predict(sample_features)
                
                # Расширяем метки на все данные с помощью ближайших соседей
                nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
                nn.fit(sample_features)
                _, indices = nn.kneighbors(features_scaled)
                labels = sample_labels[indices.flatten()]
            else:
                labels = dbscan.fit_predict(features_scaled)
            
            # Создаем спектрограмму для визуализации
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            
            # Создаем цветную карту кластеров
            time_frames = D.shape[1]
            cluster_map = np.zeros((D.shape[0], time_frames, 3), dtype=np.uint8)
            
            # Интерполируем метки кластеров
            if len(labels) != time_frames:
                x_old = np.linspace(0, 1, len(labels))
                x_new = np.linspace(0, 1, time_frames)
                labels_resampled = np.interp(x_new, x_old, labels)
                labels_resampled = labels_resampled.astype(int)
            else:
                labels_resampled = labels
            
            # Нормализуем метки для colormap
            unique_labels = np.unique(labels_resampled)
            n_clusters = len(unique_labels)
            
            if n_clusters > 0:
                # Создаем нормализованные метки
                normalized_labels = np.zeros_like(labels_resampled, dtype=np.float32)
                for i, label in enumerate(unique_labels):
                    mask = labels_resampled == label
                    if label == -1:  # Шум
                        normalized_labels[mask] = 0
                    else:
                        normalized_labels[mask] = (i + 1) * 255 / n_clusters
                
                # Применяем colormap
                for i in range(time_frames):
                    color_val = int(normalized_labels[i])
                    color = cv2.applyColorMap(
                        np.array([[color_val]], dtype=np.uint8), 
                        cv2.COLORMAP_JET
                    )[0, 0]
                    cluster_map[:, i] = color
            
            result = {
                'labels': labels,
                'cluster_map': cluster_map,
                'features': features_scaled,
                'spectrogram': D,
                'n_clusters': len(np.unique(labels[labels != -1])),
                'noise_points': np.sum(labels == -1)
            }
            
            with self._cache_lock:
                self._dbscan_cache[cache_key] = result
            
            logger.info(f"DBSCAN кластеризация выполнена успешно с {result['n_clusters']} кластерами")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при применении DBSCAN: {str(e)}", exc_info=True)
            raise

    def _estimate_optimal_clusters(self, features, max_k):
        """Оценка оптимального количества кластеров"""
        try:
            if len(features) < max_k:
                return max(2, len(features) // 2)
            
            # Используем метод локтя (упрощенная версия)
            inertias = []
            k_range = range(2, min(max_k + 1, len(features)))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(features)
                inertias.append(kmeans.inertia_)
            
            # Находим "локоть"
            if len(inertias) > 2:
                # Простой метод поиска локтя
                diffs = np.diff(inertias)
                diff2 = np.diff(diffs)
                if len(diff2) > 0:
                    elbow_idx = np.argmax(diff2) + 2  # +2 из-за двойного diff и начала с k=2
                    return k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[-1]
            
            # Возвращаем значение по умолчанию
            return min(5, max_k)
            
        except Exception as e:
            logger.warning(f"Не удалось оценить оптимальное количество кластеров: {str(e)}")
            return min(5, max_k)

    def create_cluster_visualization(self, result, output_path=None):
        """Создание визуализации кластеризации"""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Оригинальная спектрограмма
            librosa.display.specshow(
                result['spectrogram'], 
                sr=22050, 
                x_axis='time', 
                y_axis='hz',
                ax=axes[0],
                cmap='viridis'
            )
            axes[0].set_title('Оригинальная спектрограмма')
            axes[0].set_ylabel('Частота (Гц)')
            
            # Кластеризованная версия
            axes[1].imshow(
                result['cluster_map'], 
                aspect='auto', 
                origin='lower',
                extent=[0, result['cluster_map'].shape[1], 0, result['cluster_map'].shape[0]]
            )
            axes[1].set_title('Кластеризация')
            axes[1].set_xlabel('Время')
            axes[1].set_ylabel('Частота')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                return output_path
            else:
                # Возвращаем изображение в виде байтов
                img_io = io.BytesIO()
                plt.savefig(img_io, format='PNG', dpi=150, bbox_inches='tight')
                img_io.seek(0)
                plt.close(fig)
                return img_io
                
        except Exception as e:
            logger.error(f"Ошибка при создании визуализации: {str(e)}")
            raise

    def save_clustered_audio(self, y, sr, labels, output_path):
        """Сохранение кластеризованного аудио (отдельные файлы для каждого кластера)"""
        try:
            unique_labels = np.unique(labels)
            saved_files = []
            
            # Рассчитываем временные сегменты
            time_per_feature = len(y) / len(labels)
            
            for label in unique_labels:
                if label == -1:  # Пропускаем шум для DBSCAN
                    continue
                    
                # Находим индексы этого кластера
                cluster_indices = np.where(labels == label)[0]
                
                # Создаем маску для аудио
                audio_mask = np.zeros(len(y), dtype=bool)
                for idx in cluster_indices:
                    start_sample = int(idx * time_per_feature)
                    end_sample = int((idx + 1) * time_per_feature)
                    end_sample = min(end_sample, len(y))
                    audio_mask[start_sample:end_sample] = True
                
                # Извлекаем аудио для этого кластера
                cluster_audio = y * audio_mask.astype(float)
                
                # Сохраняем файл
                cluster_filename = f"{output_path}_cluster_{label}.wav"
                librosa.output.write_wav(cluster_filename, cluster_audio, sr)
                saved_files.append(cluster_filename)
            
            return saved_files
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении кластеризованного аудио: {str(e)}")
            raise
