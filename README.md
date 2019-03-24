# Sonia
Этот проект сделан в рамках програмного проекта в моем университете. Нейронная сеть написана и обучена с помощью библиотеки  pyTorch языка питон. 

Файл Instruction содержит инструкцию по запуску генератора музыки на компьютере пользователя.

Файл SoundFont содержит ссылку для скачивания нужного пакета для преобразования музыки.

Файл Generator.py содержит сам генератор, который нужно запускать на своем устройстве.

Файл decoder содержит модель обученную на композициях Чайковского.

В файле training model находится обучение одной модели на двух разных наборе данных. Первый - набор очень простой музыки, второй - 12 композиций П.И. Чайковского. Именно модель с обучения на этих данных находится в файле decoder.
