from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, send_from_directory
import os
import numpy as np
import io
import logging
import sys
from audio_processor import AudioProcessor
import uuid
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = 'static/audio_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# Создаем экземпляр AudioProcessor
audio_processor = AudioProcessor(max_workers=4)

# Глобальные переменные для текущего аудио
current_audio = None
current_sr = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("audio_index.html")

@app.route("/upload", methods=["POST"])
def upload_audio():
    global current_audio, current_sr
    
    if "audio" not in request.files:
        logger.error("Ошибка: нет файла в запросе")
        return jsonify({"error": "No file provided"}), 400

    file = request.files["audio"]
    logger.info(f"Получен файл: {file.filename}")
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Сохраняем временный файл
            filename = str(uuid.uuid4()) + "_" + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Загружаем аудио
            current_audio, current_sr = audio_processor.load_audio(filepath)
            logger.info(f"Аудио загружено успешно. Длительность: {len(current_audio)/current_sr:.2f} сек")
            
            return jsonify({
                "success": True,
                "filename": filename,
                "duration": len(current_audio)/current_sr,
                "sample_rate": current_sr
            })
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке аудио: {e}")
            return jsonify({"error": "Invalid audio file"}), 400
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route("/k_means", methods=["POST"])
def k_means():
    global current_audio, current_sr
    if current_audio is None:
        return jsonify({"error": "No audio loaded"}), 400

    try:
        k = int(request.form.get('k', 5))
        attempts = int(request.form.get('attempts', 10))
        
        logger.info(f"Применение K-means с k={k}, attempts={attempts}")
        
        # Применяем K-means кластеризацию
        result = audio_processor.apply_kmeans(current_audio, current_sr, k, attempts)
        
        # Создаем визуализацию
        img_io = audio_processor.create_cluster_visualization(result)
        
        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        logger.error(f"Ошибка при применении K-means: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route("/mean_shift", methods=["POST"])
def mean_shift():
    global current_audio, current_sr
    if current_audio is None:
        return jsonify({"error": "No audio loaded"}), 400
        
    try:
        bandwidth = float(request.form.get('bandwidth', 0.5))
        max_iterations = int(request.form.get('max_iterations', 300))
        
        logger.info(f"Применение Mean Shift с bandwidth={bandwidth}, max_iterations={max_iterations}")
        
        # Применяем Mean Shift кластеризацию
        result = audio_processor.apply_mean_shift(current_audio, current_sr, bandwidth, max_iterations)
        
        # Создаем визуализацию
        img_io = audio_processor.create_cluster_visualization(result)
        
        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        logger.error(f"Ошибка при применении Mean Shift: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route("/dbscan", methods=["POST"])
def dbscan_segmentation():
    global current_audio, current_sr
    if current_audio is None:
        return jsonify({"error": "No audio loaded"}), 400

    try:
        eps = float(request.form.get('eps', 0.5))
        min_samples = int(request.form.get('min_samples', 10))
        
        logger.info(f"Применение DBSCAN с eps={eps}, min_samples={min_samples}")
        
        # Применяем DBSCAN кластеризацию
        result = audio_processor.apply_dbscan(current_audio, current_sr, eps, min_samples)
        
        # Создаем визуализацию
        img_io = audio_processor.create_cluster_visualization(result)
        
        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        logger.error(f"Ошибка при применении DBSCAN: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route("/get_audio_info", methods=["GET"])
def get_audio_info():
    global current_audio, current_sr
    if current_audio is None:
        return jsonify({"error": "No audio loaded"}), 400
    
    try:
        info = {
            "duration": len(current_audio) / current_sr,
            "sample_rate": current_sr,
            "samples": len(current_audio),
            "channels": 1,  # librosa загружает моно по умолчанию
            "format": "float32"
        }
        return jsonify(info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/download_clustered", methods=["POST"])
def download_clustered():
    global current_audio, current_sr
    if current_audio is None:
        return jsonify({"error": "No audio loaded"}), 400

    try:
        method = request.form.get('method', 'kmeans')
        
        # Получаем результат кластеризации
        if method == 'kmeans':
            k = int(request.form.get('k', 5))
            result = audio_processor.apply_kmeans(current_audio, current_sr, k)
        elif method == 'mean_shift':
            bandwidth = float(request.form.get('bandwidth', 0.5))
            result = audio_processor.apply_mean_shift(current_audio, current_sr, bandwidth)
        elif method == 'dbscan':
            eps = float(request.form.get('eps', 0.5))
            min_samples = int(request.form.get('min_samples', 10))
            result = audio_processor.apply_dbscan(current_audio, current_sr, eps, min_samples)
        else:
            return jsonify({"error": "Unknown clustering method"}), 400
        
        # Сохраняем кластеризованные аудиофайлы
        output_base = os.path.join(app.config['UPLOAD_FOLDER'], f"clustered_{method}")
        saved_files = audio_processor.save_clustered_audio(
            current_audio, current_sr, result['labels'], output_base
        )
        
        return jsonify({
            "success": True,
            "files": [os.path.basename(f) for f in saved_files],
            "count": len(saved_files)
        })
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении кластеризованного аудио: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route("/download_file/<filename>")
def download_file(filename):
    """Скачивание сохраненного файла"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Ошибка при скачивании файла: {str(e)}")
        return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Используем другой порт, чтобы не конфликтовать с image processor
