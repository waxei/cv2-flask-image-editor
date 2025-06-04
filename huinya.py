import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.cluster import DBSCAN, KMeans
from skimage.segmentation import active_contour
import logging
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/app/static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tiff'}
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def create_upload_folder():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

def kmeans_segmentation(img, k=3):
    """Выполняет сегментацию изображения методом K-Means"""
    try:
        # Изменение размера для ускорения обработки
        small_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
        pixels = small_img.reshape(-1, 3).astype(np.float32)
        
        # Используем KMeans из sklearn для большей стабильности
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_
        
        segmented = centers[labels].reshape(small_img.shape)
        segmented = cv2.resize(segmented.astype(np.uint8), (img.shape[1], img.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        return segmented
    except Exception as e:
        logger.error(f"K-Means error: {str(e)}")
        return img

def mean_shift_segmentation(img, spatial_radius=20, color_radius=20, max_level=2):
    """Выполняет сегментацию изображения методом Mean Shift"""
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

def dbscan_segmentation(img, eps=3.0, min_samples=10):
    """Выполняет сегментацию изображения методом DBSCAN"""
    try:
        # Сильно уменьшаем изображение для DBSCAN
        small_img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        pixels = small_img.reshape(-1, 3)
        
        # Нормализуем пиксели
        pixels = pixels / 255.0
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(pixels)
        
        # Визуализация
        unique_labels = np.unique(labels)
        colors = [np.random.randint(0, 255, 3) for _ in unique_labels]
        colored = np.zeros((small_img.shape[0], small_img.shape[1], 3), dtype=np.uint8)
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # Шум
                colored[labels.reshape(small_img.shape[:2]) == label] = [0, 0, 0]
            else:
                colored[labels.reshape(small_img.shape[:2]) == label] = colors[i]
        
        # Возвращаем к исходному размеру
        return cv2.resize(colored, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        logger.error(f"DBSCAN error: {str(e)}")
        return img

def active_contour_segmentation(img, alpha=0.2, beta=0.5, resolution=100, center_x=None, center_y=None, radius=None):
    """Выполняет сегментацию изображения методом Active Contour (Snake)"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Автоматическое определение центра и радиуса, если не заданы
        if center_x is None or center_y is None:
            center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2
        if radius is None:
            radius = min(gray.shape) // 4
            
        # Генерация начального контура
        s = np.linspace(0, 2*np.pi, resolution)
        x = center_x + radius * np.cos(s)
        y = center_y + radius * np.sin(s)
        init = np.array([x, y]).T
        
        # Сегментация активного контура
        snake = active_contour(gray, init, alpha=alpha, beta=beta, gamma=0.01)
        
        # Рисуем контур на изображении
        result = img.copy()
        snake_int = np.array(snake, dtype=np.int32)
        cv2.polylines(result, [snake_int], isClosed=True, color=(0, 255, 0), thickness=2)
        return result
    except Exception as e:
        logger.error(f"Active Contour error: {str(e)}")
        return img

def watersheeld(img):   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Удаление шума
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Определение фона
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Определение переднего плана
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Нахождение неизвестной области
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Маркировка компонентов
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    # Применение алгоритма водораздела
    markers = cv2.watershed(img, markers)

    mask = np.zeros_like(markers, dtype=np.uint8)
    # mask[markers > 1] = 255 
    mask[markers <= 1] = 255 

    # Удаление всего, что не вошло в зону выделенного объекта
    result = cv2.bitwise_and(img, img, mask=mask)

    # Окрашивание границ (опционально)
    result[markers == -1] = [102, 0, 153]
    return result



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/app/static/uploads/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            create_upload_folder()
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            return redirect(url_for('edit_image', filename=unique_filename))
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return redirect(request.url)
    
    return redirect(request.url)

@app.route('/edit/<filename>')
def edit_image(filename):
    return render_template('edit.html', filename=filename)

@app.route('/process', methods=['POST'])
def process_image():
    filename = request.form.get('filename')
    if not filename:
        return redirect(url_for('index'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return redirect(url_for('index'))
    
    try:
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError("Failed to load image")
        
        # Базовые операции
        if 'resize' in request.form:
            width = int(request.form.get('width', img.shape[1]))
            height = int(request.form.get('height', img.shape[0]))
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        
        if 'crop' in request.form:
            x = int(request.form.get('x', 0))
            y = int(request.form.get('y', 0))
            w = int(request.form.get('w', img.shape[1]))
            h = int(request.form.get('h', img.shape[0]))
            img = img[y:y+h, x:x+w]
        
        if 'flip' in request.form:
            flip_code = int(request.form['flip_code'])
            img = cv2.flip(img, flip_code)

        if 'rotate' in request.form:
            angle = float(request.form['angle'])
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

        if 'brightness_contrast' in request.form:
            alpha = float(request.form['alpha'])
            beta = int(request.form['beta'])
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if 'color_balance' in request.form:
            b, g, r = cv2.split(img)
            b = cv2.add(b, int(request.form['blue']))
            g = cv2.add(g, int(request.form['green']))
            r = cv2.add(r, int(request.form['red']))
            img = cv2.merge((b, g, r))

        if 'add_noise' in request.form:
            noise_type = request.form['noise_type']
            if noise_type == 'gaussian':
                mean = 0
                var = 100
                sigma = var ** 0.5
                gauss = np.random.normal(mean, sigma, img.shape)
                img = cv2.add(img, gauss.astype(np.uint8))
            elif noise_type == 's&p':
                s_vs_p = 0.5
                amount = 0.004
                out = np.copy(img)
                num_salt = np.ceil(amount * img.size * s_vs_p)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
                out[coords] = 255
                num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
                out[coords] = 0
                img = out

        if 'blur' in request.form:
            blur_type = request.form['blur_type']
            if blur_type == 'average':
                img = cv2.blur(img, (5, 5))
            elif blur_type == 'gaussian':
                img = cv2.GaussianBlur(img, (5, 5), 0)
            elif blur_type == 'median':
                img = cv2.medianBlur(img, 5)

        if 'color_space' in request.form:
            color_space = request.form['color_space']
            if color_space == 'hsv':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif color_space == 'gray':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if 'find_object' in request.form:
            color = request.form['color']
            color = list(map(int, color.split(',')))
            if len(color) == 3:
                lower = np.array(color) - 10
                upper = np.array(color) + 10
                mask = cv2.inRange(img, lower, upper)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Операции сегментации
        if 'kmeans' in request.form:
            k = int(request.form.get('k', 3))
            img = kmeans_segmentation(img, k)
        
        if 'mean_shift' in request.form:
            spatial_radius = int(request.form.get('spatial_radius', 20))
            color_radius = int(request.form.get('color_radius', 20))
            max_level = int(request.form.get('max_level', 2))
            img = mean_shift_segmentation(img, spatial_radius, color_radius, max_level)
        
        if 'dbscan' in request.form:
            eps = float(request.form.get('eps', 3.0))
            min_samples = int(request.form.get('min_samples', 10))
            img = dbscan_segmentation(img, eps, min_samples)
        
        if 'active_contour' in request.form:
            alpha = float(request.form.get('alpha', 0.2))
            beta = float(request.form.get('beta', 0.5))
            resolution = int(request.form.get('resolution', 100))
            center_x = int(request.form.get('center_x', img.shape[1]//2))
            center_y = int(request.form.get('center_y', img.shape[0]//2))
            radius = int(request.form.get('radius', min(img.shape)//4))
            img = active_contour_segmentation(img, alpha, beta, resolution, center_x, center_y, radius)

        if 'watershed' in request.form:
            img = watersheeld(img)
        
        # Сохранение результата
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        processed_filename = f"processed_{timestamp}_{filename}"
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        cv2.imwrite(processed_filepath, img)
        
        return redirect(url_for('edit_image', filename=processed_filename))
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return redirect(url_for('edit_image', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    create_upload_folder()
    app.run(debug=True)



    