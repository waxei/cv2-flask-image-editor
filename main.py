from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
import cv2
import numpy as np
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Для отображения сообщений flash

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(Exception)
def handle_error(e):
    flash(f"Произошла ошибка: {str(e)}")
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash("Файл не найден!")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash("Файл не выбран!")
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash("Недопустимый формат файла!")
        return redirect(request.url)

    original_filename = file.filename
    file_extension = original_filename.rsplit('.', 1)[1].lower()  # Получаем расширение файла
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{timestamp}_{uuid.uuid4().hex}.{file_extension}"

    # Сохранение файла с уникальным именем
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    return redirect(url_for('editor', filename=unique_filename))

@app.route('/edit/<action>/<filename>', methods=['POST'])
def edit_image(action, filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        flash("Файл не найден!")
        return redirect(url_for('index'))

    img = cv2.imread(filepath)
    if img is None:
        flash("Ошибка чтения изображения!")
        return redirect(url_for('index'))

    try:
        if action == "resize":
            # Изменение размера
            scale = request.form.get("scale")
            width = request.form.get("width")
            height = request.form.get("height")
            interpolation = request.form.get("interpolation", "auto")
            
            original_h, original_w = img.shape[:2]
            
            if scale:
                scale_factor = float(scale)
                new_w = int(original_w * scale_factor)
                new_h = int(original_h * scale_factor)
            else:
                new_w = int(width) if width else original_w
                new_h = int(height) if height else original_h

            if interpolation == "auto":
                if new_w > original_w or new_h > original_h:
                    interp = cv2.INTER_CUBIC
                else:
                    interp = cv2.INTER_AREA
            else:
                interp_map = {
                    "nearest": cv2.INTER_NEAREST,
                    "bilinear": cv2.INTER_LINEAR,
                    "bicubic": cv2.INTER_CUBIC
                }
                interp = interp_map.get(interpolation, cv2.INTER_LINEAR)

            img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        elif action == "convert_color_space":
            color_space = request.form.get("color_space")
            if color_space == "hsv":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif color_space == "grayscale":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif color_space == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                flash("Недопустимое цветовое пространство!")
                return redirect(url_for('editor', filename=filename))
        
        elif action == "find_object_by_color":
            color_space = request.form.get("color_space")
            color = list(map(int, request.form.get("color").split(',')))  # Получаем цвет в формате "R,G,B" или "H,S,V"
            tolerance = int(request.form.get("tolerance", 10))  # Допуск для поиска цвета
            action_type = request.form.get("action_type")  # "bounding_box" или "crop"

            if color_space == "hsv":
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                lower_bound = np.array([max(0, color[0] - tolerance), max(0, color[1] - tolerance), max(0, color[2] - tolerance)])
                upper_bound = np.array([min(179, color[0] + tolerance), min(255, color[1] + tolerance), min(255, color[2] + tolerance)])
                mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
            else:  # RGB
                lower_bound = np.array([max(0, color[0] - tolerance), max(0, color[1] - tolerance), max(0, color[2] - tolerance)])
                upper_bound = np.array([min(255, color[0] + tolerance), min(255, color[1] + tolerance), min(255, color[2] + tolerance)])
                mask = cv2.inRange(img, lower_bound, upper_bound)

            # Нахождение контуров объекта
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])  # Координаты ограничивающей рамки

                if action_type == "bounding_box":
                    # Рисуем ограничивающую рамку
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                elif action_type == "crop":
                    # Обрезаем изображение по координатам объекта
                    img = img[y:y + h, x:x + w]
            else:
                flash("Объект с указанным цветом не найден!")
                return redirect(url_for('editor', filename=filename))

        elif action == "crop":
            # Вырезка фрагмента
            if 'mask' in request.files:
                mask_file = request.files['mask']
                if mask_file.filename != '':
                    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + filename)
                    mask_file.save(mask_path)
                    mask = cv2.imread(mask_path, 0)
                    os.remove(mask_path)
                    
                    if mask.shape != img.shape[:2]:
                        flash("Маска должна соответствовать размеру изображения!")
                        return redirect(url_for('editor', filename=filename))
                    
                    img = cv2.bitwise_and(img, img, mask=mask)
            else:
                x = int(request.form.get("x", 0))
                y = int(request.form.get("y", 0))
                w = int(request.form.get("w", img.shape[1]))
                h = int(request.form.get("h", img.shape[0]))
                
                if x + w > img.shape[1] or y + h > img.shape[0]:
                    flash("Некорректные координаты обрезки!")
                    return redirect(url_for('editor', filename=filename))

                img = img[y:y+h, x:x+w]

        elif action == "mirror":
            # Зеркальное отражение
            direction = request.form.get("direction")
            flip_code = 1 if direction == "horizontal" else 0 if direction == "vertical" else -1
            img = cv2.flip(img, flip_code)

        elif action == "rotate":
            # Поворот изображения
            angle = float(request.form.get("angle", 0))
            center_x = request.form.get("center_x")
            center_y = request.form.get("center_y")
            
            h, w = img.shape[:2]
            center = (w//2, h//2) if not (center_x and center_y) else (int(center_x), int(center_y))
            
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

        elif action == "brightness_contrast":
            # Яркость и контраст
            brightness = int(request.form.get("brightness", 0))
            contrast = float(request.form.get("contrast", 1.0))
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

        elif action == "color_balance":
            # Цветовой баланс
            b = float(request.form.get("b", 1.0))
            g = float(request.form.get("g", 1.0))
            r = float(request.form.get("r", 1.0))
            
            img = img.astype(np.float32)
            img[:,:,0] = np.clip(img[:,:,0] * b, 0, 255)
            img[:,:,1] = np.clip(img[:,:,1] * g, 0, 255)
            img[:,:,2] = np.clip(img[:,:,2] * r, 0, 255)
            img = img.astype(np.uint8)

        elif action == "noise":
            # Добавление шума
            noise_type = request.form.get("type")
            if noise_type == "gaussian":
                sigma = int(request.form.get("sigma", 25))
                noise = np.random.normal(0, sigma, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)
            elif noise_type == "salt_pepper":
                prob = float(request.form.get("prob", 0.02))
                noise_img = img.copy()
                black = np.array([0, 0, 0], dtype=np.uint8)
                white = np.array([255, 255, 255], dtype=np.uint8)
                probs = np.random.random(img.shape[:2])
                noise_img[probs < prob] = black
                noise_img[probs > 1 - prob] = white
                img = noise_img

        elif action == "blur":
            # Размытие
            kernel_size = int(request.form.get("size", 5))
            blur_type = request.form.get("type")
            
            if blur_type == "average":
                img = cv2.blur(img, (kernel_size, kernel_size))
            elif blur_type == "gaussian":
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            elif blur_type == "median":
                img = cv2.medianBlur(img, kernel_size)

        # Сохранение изменений
        cv2.imwrite(filepath, img)

    except Exception as e:
        flash(f"Ошибка обработки изображения: {str(e)}")
        return redirect(url_for('editor', filename=filename))

    return redirect(url_for('editor', filename=filename))

@app.route('/editor/<filename>')
def editor(filename):
    return render_template('editor.html', filename=filename)

@app.route('/save/<filename>')
def save_image(filename):
    try:
        # Путь к файлу
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Возвращаем файл для скачивания
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        flash(f"Ошибка сохранения: {str(e)}")
        return redirect(url_for('editor', filename=filename))

@app.route('/delete_file/<filename>')
def delete_file(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return f"Файл {filename} удален", 200
        else:
            return f"Файл {filename} не найден", 404
    except Exception as e:
        return f"Ошибка при удалении файла: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
