
# Парсер данных с изображения из сайта для трейдинга
 Программа позволяет собирать данные из типового изображения с графиком цены актива.
 Из изображения собирается информация о тикере и ценовых метках на 
 вертикальной шкале графика. 
 
## Установка
### Установка пакетов
```
cd src
pip install -r requirements.txt
```
### Установка tesseract-ocr
Для Linux:
```
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```
Для Windows:

1. https://github.com/UB-Mannheim/tesseract/wiki
2. Установить в `src/config.py` корректный путь к файлу c программой в переменной `windows_tesseract_path`.
По умолчанию равно `D:\Program Files\Tesseract-OCR\tesseract.exe`.

## Парсинг

Запуск парсинга с указанием пути к изображением:
```
python src/main.py --path path/to/file.jpg
```
Запуск парсинга с указанием ссылки на изображение:
```
python src/main.py --url https://link/to/file.jpg
```
Запуск с указанием пути к исходному изображению и пути к файлу с результатом (по умалчанию - `output.jpg`):
```
python src/main.py --path path/to/file.jpg --output path/to/output.json
```

## Демо

 Для тестирования можно запустить файл `debugging_and_testing.py`, который обработает все изображения из 
 папки `original_images` и сохранит их в папку `results` со специальными метками на изображении, 
 на которых будут указаны результаты парсинга. Во время тестирования будет также вывоится окна с обработанными изображениями
 для большей наглядности.

