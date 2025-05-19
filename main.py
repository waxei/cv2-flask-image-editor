from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
import cv2
import numpy as np
import uuid
from datetime import datetime
import logging
import sys
from image_processor import ImageProcessor
import threading

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Для отображения сообщений flash

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff'}

# Создаем экземпляр ImageProcessor
image_processor = ImageProcessor(max_workers=4)

# Кэш для изображений
_image_cache = {}
_cache_lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_cached_image(filepath):
    """Получение изображения из кэша или загрузка из файла"""
    with _cache_lock:
        if filepath in _image_cache:
            return _image_cache[filepath].copy()
        
        img = cv2.imread(filepath)
        if img is not None:
            _image_cache[filepath] = img.copy()
        return img

def save_image_to_cache(filepath, img):
    """Сохранение изображения в кэш"""
    with _cache_lock:
        _image_cache[filepath] = img.copy()

@app.errorhandler(Exception)
def handle_error(e):
    logger.error(f"Произошла ошибка: {str(e)}", exc_info=True)
    flash(f"Произошла ошибка: {str(e)}")
    return redirect(url_for('index'))

@app.route('/')
def index():
    logger.info("Открыта главная страница")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    logger.info("Начало загрузки изображения")
    if 'file' not in request.files:
        logger.warning("Файл не найден в запросе")
        flash("Файл не найден!")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        logger.warning("Имя файла пустое")
        flash("Файл не выбран!")
        return redirect(request.url)

    if not allowed_file(file.filename):
        logger.warning(f"Недопустимый формат файла: {file.filename}")
        flash("Недопустимый формат файла!")
        return redirect(request.url)

    original_filename = file.filename
    file_extension = original_filename.rsplit('.', 1)[1].lower()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{timestamp}_{uuid.uuid4().hex}.{file_extension}"

    logger.info(f"Загрузка файла: {original_filename} -> {unique_filename}")
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    # Загружаем изображение в кэш
    img = cv2.imread(filepath)
    if img is not None:
        save_image_to_cache(filepath, img)
    
    logger.info(f"Файл успешно сохранен: {filepath}")

    return redirect(url_for('editor', filename=unique_filename))

@app.route('/edit/<action>/<filename>', methods=['POST'])
def edit_image(action, filename):
    logger.info(f"Начало обработки изображения: {action} для файла {filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        logger.error(f"Файл не найден: {filepath}")
        flash("Файл не найден!")
        return redirect(url_for('index'))

    # Получаем изображение из кэша
    img = get_cached_image(filepath)
    if img is None:
        logger.error(f"Ошибка чтения изображения: {filepath}")
        flash("Ошибка чтения изображения!")
        return redirect(url_for('index'))

    try:
        # Обработка изображения в зависимости от действия
        if action == "resize":
            scale = request.form.get("scale")
            width = request.form.get("width")
            height = request.form.get("height")
            interpolation = request.form.get("interpolation", "auto")
            
            img = image_processor.resize_image(img, scale, width, height, interpolation)
            logger.info("Изменение размера выполнено успешно")

        elif action == "convert_color_space":
            color_space = request.form.get("color_space")
            img = image_processor.convert_color_space(img, color_space)
            logger.info(f"Конвертация в цветовое пространство {color_space} выполнена успешно")
        
        elif action == "find_object_by_color":
            color_space = request.form.get("color_space")
            color = list(map(int, request.form.get("color").split(',')))
            tolerance = int(request.form.get("tolerance", 10))
            action_type = request.form.get("action_type")
            
            img = image_processor.find_object_by_color(img, color_space, color, tolerance, action_type)
            logger.info("Поиск объекта по цвету выполнен успешно")

        elif action == "crop":
            if 'mask' in request.files:
                mask_file = request.files['mask']
                if mask_file.filename != '':
                    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + filename)
                    mask_file.save(mask_path)
                    mask = cv2.imread(mask_path, 0)
                    os.remove(mask_path)
                    
                    img = image_processor.crop_with_mask(img, mask)
            else:
                x = int(request.form.get("x", 0))
                y = int(request.form.get("y", 0))
                w = int(request.form.get("w", img.shape[1]))
                h = int(request.form.get("h", img.shape[0]))
                
                img = image_processor.crop_image(img, x, y, w, h)
            logger.info("Обрезка изображения выполнена успешно")

        elif action == "mirror":
            direction = request.form.get("direction")
            img = image_processor.mirror_image(img, direction)
            logger.info(f"Зеркальное отражение по направлению {direction} выполнено успешно")

        elif action == "rotate":
            angle = float(request.form.get("angle", 0))
            center_x = request.form.get("center_x")
            center_y = request.form.get("center_y")
            
            center = None
            if center_x and center_y:
                center = (int(center_x), int(center_y))
            
            img = image_processor.rotate_image(img, angle, center)
            logger.info(f"Поворот на угол {angle} выполнен успешно")

        elif action == "brightness_contrast":
            brightness = int(request.form.get("brightness", 0))
            contrast = float(request.form.get("contrast", 1.0))
            img = image_processor.apply_brightness_contrast(img, brightness, contrast)
            logger.info("Корректировка яркости и контраста выполнена успешно")

        elif action == "color_balance":
            b = float(request.form.get("b", 1.0))
            g = float(request.form.get("g", 1.0))
            r = float(request.form.get("r", 1.0))
            
            img = image_processor.apply_color_balance(img, b, g, r)
            logger.info("Корректировка цветового баланса выполнена успешно")

        elif action == "noise":
            noise_type = request.form.get("type")
            params = {}
            
            if noise_type == "gaussian":
                params["sigma"] = int(request.form.get("sigma", 25))
            elif noise_type == "salt_pepper":
                params["prob"] = float(request.form.get("prob", 0.02))
            
            img = image_processor.add_noise(img, noise_type, **params)
            logger.info(f"Добавление шума типа {noise_type} выполнено успешно")

        elif action == "blur":
            kernel_size = int(request.form.get("size", 5))
            blur_type = request.form.get("type")
            img = image_processor.apply_blur(img, kernel_size, blur_type)
            logger.info(f"Применение размытия типа {blur_type} выполнено успешно")

        elif action == "watershed":
            blur_size = int(request.form.get("blur_size", 5))
            min_distance = int(request.form.get("min_distance", 10))
            img = image_processor.apply_watershed(img, blur_size, min_distance)
            logger.info("Сегментация методом водораздела выполнена успешно")

        elif action == "mean_shift":
            spatial_radius = int(request.form.get("spatial_radius", 20))
            color_radius = int(request.form.get("color_radius", 20))
            max_level = int(request.form.get("max_level", 2))
            img = image_processor.apply_mean_shift(img, spatial_radius, color_radius, max_level)
            logger.info("Сегментация методом Mean Shift выполнена успешно")

        elif action == "kmeans":
            k = int(request.form.get("k", 3))
            attempts = int(request.form.get("attempts", 10))
            img = image_processor.apply_kmeans(img, k, attempts)
            logger.info("Сегментация методом K-means выполнена успешно")

        elif action == "dbscan":
            eps = float(request.form.get("eps", 10))
            min_samples = int(request.form.get("min_samples", 5))
            img = image_processor.apply_dbscan(img, eps, min_samples)
            logger.info("Сегментация методом DBSCAN выполнена успешно")

        elif action == "active_contour":
            alpha = float(request.form.get("alpha", 0.015))
            beta = float(request.form.get("beta", 10))
            gamma = float(request.form.get("gamma", 0.001))
            img = image_processor.apply_active_contour(img, alpha, beta, gamma)
            logger.info("Сегментация активным контуром выполнена успешно")

        elif action == "binarize":
            method = request.form.get("method", "global")
            threshold = request.form.get("threshold")
            block_size = request.form.get("block_size")
            C = request.form.get("C")
            
            if threshold:
                threshold = int(threshold)
            if block_size:
                block_size = int(block_size)
            if C:
                C = int(C)
            
            img = image_processor.apply_binarize(img, method, threshold, block_size, C)
            logger.info(f"Бинаризация методом {method} выполнена успешно")

        elif action == "threshold":
            method = request.form.get("method", "binary")
            threshold = int(request.form.get("threshold", 127))
            max_value = int(request.form.get("max_value", 255))
            img = image_processor.apply_threshold(img, method, threshold, max_value)
            logger.info(f"Пороговая обработка методом {method} выполнена успешно")

        elif action == "sobel_edges":
            direction = request.form.get("direction", "combined")
            ksize = int(request.form.get("ksize", 3))
            img = image_processor.apply_sobel_edges(img, direction, ksize)
            logger.info(f"Выделение краев Собеля по направлению {direction} выполнено успешно")

        elif action == "canny_edges":
            low_thresh = int(request.form.get("low_thresh", 100))
            high_thresh = int(request.form.get("high_thresh", 200))
            img = image_processor.apply_canny_edges(img, low_thresh, high_thresh)
            logger.info("Выделение краев Канни выполнено успешно")

        # Сохраняем обработанное изображение
        cv2.imwrite(filepath, img)
        save_image_to_cache(filepath, img)
        logger.info(f"Изображение успешно сохранено: {filepath}")
        
        return redirect(url_for('editor', filename=filename))

    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}", exc_info=True)
        flash(f"Ошибка при обработке изображения: {str(e)}")
        return redirect(url_for('editor', filename=filename))

@app.route('/editor/<filename>')
def editor(filename):
    return render_template('editor.html', filename=filename)

@app.route('/save/<filename>')
def save_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete_file/<filename>')
def delete_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        # Удаляем из кэша
        with _cache_lock:
            if filepath in _image_cache:
                del _image_cache[filepath]
        logger.info(f"Файл удален: {filepath}")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
