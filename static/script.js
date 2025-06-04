document.addEventListener('DOMContentLoaded', function() {
    // Получаем все кнопки действий
    const actionButtons = document.querySelectorAll('.toolbar button[data-action]');
    
    // Получаем все формы
    const forms = document.querySelectorAll('.form-group, .clustering-method');
    
    // Получаем оверлей загрузки
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    // Функция для скрытия всех форм
    function hideAllForms() {
        forms.forEach(form => {
            form.style.display = 'none';
        });
    }
    
    // Обработчик клика по кнопкам
    actionButtons.forEach(button => {
        button.addEventListener('click', function() {
            const action = this.getAttribute('data-action');
            
            // Скрываем все формы
            hideAllForms();
            
            // Показываем нужную форму
            const form = document.getElementById(`${action}-form`);
            if (form) {
                form.style.display = 'block';
            }
        });
    });
    
    // Обработчик отправки форм
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Показываем индикатор загрузки
            loadingOverlay.style.display = 'flex';
            
            // Создаем FormData из формы
            const formData = new FormData(this);
            
            // Отправляем AJAX запрос
            fetch(this.action, {
                method: 'POST',
                body: formData
            })
            .then(response => {
                // Скрываем индикатор загрузки при любом ответе
                loadingOverlay.style.display = 'none';
                if (response.redirected) {
                    window.location.href = response.url;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                loadingOverlay.style.display = 'none';
                alert('Произошла ошибка при обработке изображения');
            });
        });
    });

    // Обработчик для кнопки бинаризации
    document.querySelector('button[data-action="binarize"]').addEventListener('click', function() {
        hideAllForms();
        document.getElementById('binarize-form').style.display = 'block';
    });

    // Обработчик для кнопки кластеризации
    document.querySelector('button[data-action="clustering"]').addEventListener('click', function() {
        hideAllForms();
        document.getElementById('clustering-form').style.display = 'block';
    });

    // Обработчик выбора метода кластеризации
    document.getElementById('clustering-method').addEventListener('change', function() {
        // Скрываем все формы методов кластеризации
        document.querySelectorAll('.clustering-method').forEach(form => {
            form.style.display = 'none';
        });
        
        // Показываем выбранную форму
        const selectedMethod = this.value;
        if (selectedMethod) {
            document.getElementById(`${selectedMethod}-form`).style.display = 'block';
        }
    });
});