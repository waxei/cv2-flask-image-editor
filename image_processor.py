import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans  # add KMeans import
from sklearn.neighbors import NearestCentroid
from skimage.segmentation import active_contour
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache_lock = threading.Lock()
        self._watershed_cache = {}
        self._kmeans_cache = {}
        self._color_cache = {}

    def __del__(self):
        self.executor.shutdown()

    @staticmethod
    @lru_cache(maxsize=32)
    def get_interpolation_method(interpolation, new_size, original_size):
        if interpolation == "auto":
            if new_size[0] > original_size[0] or new_size[1] > original_size[1]:
                return cv2.INTER_CUBIC
            return cv2.INTER_AREA
        
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC
        }
        return interp_map.get(interpolation, cv2.INTER_LINEAR)

    def _process_watershed_markers(self, markers_array, img_shape_with_channels):
        result = np.zeros(img_shape_with_channels, dtype=np.uint8)
        unique_marker_values = np.unique(markers_array)
        colors = np.random.randint(0, 255, size=(len(unique_marker_values), 3), dtype=np.uint8)
        
        for i, marker_val in enumerate(unique_marker_values):
            if marker_val > 1:
                mask = markers_array == marker_val
                result[mask] = colors[i]
        
        return result

    def resize_image(self, img, scale=None, width=None, height=None, interpolation="auto"):
        original_h, original_w = img.shape[:2]
        
        if scale:
            scale_factor = float(scale)
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)
        else:
            new_w = int(width) if width else original_w
            new_h = int(height) if height else original_h

        interp = self.get_interpolation_method(interpolation, (new_w, new_h), (original_w, original_h))
        
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    def convert_color_space(self, img, color_space):
        color_space_map = {
            "hsv": cv2.COLOR_BGR2HSV,
            "grayscale": cv2.COLOR_BGR2GRAY,
            "rgb": cv2.COLOR_BGR2RGB
        }
        
        if color_space not in color_space_map:
            raise ValueError(f"Недопустимое цветовое пространство: {color_space}")
            
        return cv2.cvtColor(img, color_space_map[color_space])

    def find_object_by_color(self, img, color_space, color, tolerance, action_type):
        color_np = np.array(color, dtype=np.uint8)
        
        if color_space == "hsv":
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([max(0, color_np[0] - tolerance), max(0, color_np[1] - tolerance), max(0, color_np[2] - tolerance)])
            upper_bound = np.array([min(179, color_np[0] + tolerance), min(255, color_np[1] + tolerance), min(255, color_np[2] + tolerance)])
            mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
        else:
            lower_bound = np.array([max(0, color_np[0] - tolerance), max(0, color_np[1] - tolerance), max(0, color_np[2] - tolerance)])
            upper_bound = np.array([min(255, color_np[0] + tolerance), min(255, color_np[1] + tolerance), min(255, color_np[2] + tolerance)])
            mask = cv2.inRange(img, lower_bound, upper_bound)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("Объект с указанным цветом не найден")

        x, y, w, h = cv2.boundingRect(contours[0])
        
        if action_type == "bounding_box":
            result = img.copy()
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return result
        elif action_type == "crop":
            return img[y:y + h, x:x + w]
        else:
            raise ValueError(f"Недопустимый тип действия: {action_type}")

    def apply_brightness_contrast(self, img, brightness, contrast):
        return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    def apply_color_balance(self, img, b, g, r):
        img_float = img.astype(np.float32)
        img_float[:,:,0] = np.clip(img_float[:,:,0] * b, 0, 255)
        img_float[:,:,1] = np.clip(img_float[:,:,1] * g, 0, 255)
        img_float[:,:,2] = np.clip(img_float[:,:,2] * r, 0, 255)
        return img_float.astype(np.uint8)

    def add_noise(self, img, noise_type, **params):
        if noise_type == "gaussian":
            sigma = params.get("sigma", 25)
            noise = np.random.normal(0, sigma, img.shape).astype(np.uint8)
            return cv2.add(img, noise)
        elif noise_type == "salt_pepper":
            prob = params.get("prob", 0.02)
            result = img.copy()
            black = np.array([0, 0, 0], dtype=np.uint8)
            white = np.array([255, 255, 255], dtype=np.uint8)
            
            salt_mask = np.random.random(img.shape[:2]) < prob/2
            pepper_mask = np.random.random(img.shape[:2]) < prob/2
            
            result[salt_mask] = white
            result[pepper_mask] = black
            
            return result
        else:
            raise ValueError(f"Недопустимый тип шума: {noise_type}")

    def rotate_image(self, img, angle, center=None):
        h, w = img.shape[:2]
        if center is None:
            center = (w//2, h//2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    def mirror_image(self, img, direction):
        flip_code = 1 if direction == "horizontal" else 0 if direction == "vertical" else -1
        return cv2.flip(img, flip_code)

    def crop_image(self, img, x, y, w, h):
        if x + w > img.shape[1] or y + h > img.shape[0]:
            raise ValueError("Некорректные координаты обрезки")
        return img[y:y+h, x:x+w]

    def crop_with_mask(self, img, mask):
        if mask.shape != img.shape[:2]:
            raise ValueError("Маска должна соответствовать размеру изображения")
        return cv2.bitwise_and(img, img, mask=mask)

    def apply_blur(self, img, kernel_size, blur_type):
        if blur_type == "average":
            return cv2.blur(img, (kernel_size, kernel_size))
        elif blur_type == "gaussian":
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif blur_type == "median":
            return cv2.medianBlur(img, kernel_size)
        else:
            raise ValueError(f"Недопустимый тип размытия: {blur_type}")

    def apply_watershed(self, img, blur_size, min_distance):
        cache_key = (hash(img.tobytes()), blur_size, min_distance)
        with self._cache_lock:
            if cache_key in self._watershed_cache:
                return self._watershed_cache[cache_key]

        scale_factor = min(1.0, 1000 / max(img.shape[:2]))
        if scale_factor < 1.0:
            small_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
        else:
            small_img = img.copy()

        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, min_distance, 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(thresh, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(small_img, markers)
        
        result = self._process_watershed_markers(markers, small_img.shape)
        result[markers == -1] = [0, 0, 255]
        
        if scale_factor < 1.0:
            result = cv2.resize(result, (img.shape[1], img.shape[0]))
        
        with self._cache_lock:
            self._watershed_cache[cache_key] = result
        
        return result

    def _preprocess_image(self, img, max_size=800):
        height, width = img.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        img = img.astype(np.float32) / 255.0
        return img

    def apply_mean_shift(self, img, spatial_radius=10, color_radius=10, max_level=3):
        """Выполняет сегментацию изображения методом Mean Shift (huinya logic)"""
        try:
            # Уменьшаем изображение для ускорения обработки
            small_img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
            img_lab = cv2.cvtColor(small_img, cv2.COLOR_BGR2LAB)
            segmented = cv2.pyrMeanShiftFiltering(img_lab, spatial_radius, color_radius, max_level)
            segmented = cv2.cvtColor(segmented, cv2.COLOR_LAB2BGR)
            # Возвращаем к исходному размеру
            return cv2.resize(segmented, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            logger.error(f"Mean Shift error: {str(e)}")
            return img

    def apply_kmeans(self, img, k=5, attempts=5):
        """Выполняет сегментацию изображения методом K-Means (huinya logic)"""
        try:
            # Изменение размера для ускорения обработки
            small_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
            pixels = small_img.reshape(-1, 3).astype(np.float32)
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_
            segmented = centers[labels].reshape(small_img.shape)
            segmented = cv2.resize(segmented.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            return segmented
        except Exception as e:
            logger.error(f"K-Means error: {str(e)}")
            return img

    def apply_dbscan(self, img, eps=0.5, min_samples=10):
        """Выполняет сегментацию изображения методом DBSCAN с масштабированием признаков"""
        # Уменьшаем изображение для ускорения
        small_img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        # Подготавливаем признаки
        pixels = small_img.reshape(-1, 3).astype(np.float32)
        scaler = StandardScaler()
        features = scaler.fit_transform(pixels)
        # Кластеризация DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)
        unique_labels = np.unique(labels)
        # Присваиваем цвета
        colors = {label: (np.array([0,0,0]) if label == -1 else np.random.randint(0,255,3)) for label in unique_labels}
        # Рисуем карту кластеров
        h, w = small_img.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        label_map = labels.reshape(h, w)
        for label, color in colors.items():
            mask = (label_map == label)
            colored[mask] = color
        # Возвращаем к исходному размеру
        return cv2.resize(colored, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    def _get_cached_color(self, label):
        with self._cache_lock:
            if label not in self._color_cache:
                self._color_cache[label] = np.random.randint(0, 255, 3, dtype=np.uint8)
            return self._color_cache[label]

    def apply_active_contour(self, img, alpha, beta, gamma):
        scale_factor = min(1.0, 800 / max(img.shape[:2]))
        if scale_factor < 1.0:
            small_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
        else:
            small_img = img.copy()

        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        s = np.linspace(0, 2*np.pi, 200)
        r = h/4
        x_coords = w/2 + r*np.cos(s)
        y_coords = h/2 + r*np.sin(s)
        init = np.array([x_coords, y_coords]).T # Corrected variable names
        
        snake = active_contour(gray, init, alpha=alpha, beta=beta, gamma=gamma)
        
        result = small_img.copy()
        snake = snake.astype(np.int32)
        cv2.polylines(result, [snake], True, (0, 0, 255), 2)
        
        if scale_factor < 1.0:
            result = cv2.resize(result, (img.shape[1], img.shape[0]))
        
        return result

    def apply_binarize(self, img, method, threshold=None, block_size=None, C=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if method == "global":
            _, result = cv2.threshold(gray, threshold or 127, 255, cv2.THRESH_BINARY)
        elif method == "otsu":
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive":
            result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, block_size or 11, C or 2)
        else:
            raise ValueError(f"Недопустимый метод бинаризации: {method}")
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def apply_threshold(self, img, method, threshold, max_value):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        method_map = {
            "binary": cv2.THRESH_BINARY,
            "binary_inv": cv2.THRESH_BINARY_INV,
            "trunc": cv2.THRESH_TRUNC,
            "tozero": cv2.THRESH_TOZERO,
            "tozero_inv": cv2.THRESH_TOZERO_INV,
            "otsu": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            "triangle": cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        }
        
        if method not in method_map:
            raise ValueError(f"Недопустимый метод пороговой обработки: {method}")
        
        if method in ["otsu", "triangle"]:
            _, result = cv2.threshold(gray, 0, max_value, method_map[method])
        else:
            _, result = cv2.threshold(gray, threshold, max_value, method_map[method])
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def apply_sobel_edges(self, img, direction, ksize):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if direction == "x":
            edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        elif direction == "y":
            edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        else:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            edges = np.sqrt(sobelx**2 + sobely**2)
        
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def apply_canny_edges(self, img, low_thresh, high_thresh):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_thresh, high_thresh)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)