# Архитектор (Architector) 🧠

[English version below | Английская версия ниже](#english)

---

## 🇷🇺 Русская версия

### Описание проекта

**Архитектор** — это экспериментальная система машинного обучения, которая моделирует эмоциональный интеллект и динамику личности через волновые фазовые состояния нейронной сети. Проект объединяет классические методы глубокого обучения с инновационными подходами квантоподобных вычислений, волновой динамики и фазовой архитектуры для обработки эмоциональных данных и естественного языка.

### Основные возможности

- 🌊 **Волновая нейронная сеть** с многоуровневой фазовой динамикой
- 🎭 **Обработка эмоций** — классификация и анализ эмоциональных состояний из текста
- 🧪 **Гибридная архитектура** — комбинация традиционного backpropagation и фазовых вычислений
- 🔬 **Квантоподобные элементы** — имитация квантовых явлений в классической нейронной сети
- 💾 **Динамическое сохранение моделей** — архивирование фазовых состояний
- 📊 **Расширенная визуализация** — фазовые карты, матрицы сознания, 3D визуализации
- 🖼️ **Мультимодальная обработка** — текст, изображения и битовые сетки
- 🔄 **Режимы обучения и инференса** — гибкие конфигурации для различных сценариев

---

## 🏗️ Архитектурные компоненты и технологии

### 1. **Волновая нейронная сеть (Wave Network)**

Основной компонент проекта, расположенный в `wave_network_integrated.py`.

**Технология:**
- Обработка данных через фазовое пространство вместо традиционного векторного представления
- Каждый нейрон оперирует фазовыми углами, представляющими состояние системы
- Многоуровневая архитектура с цепочкой волновых слоёв
- Синтез локальных и глобальных фазовых полей для комплексной обработки

**Ключевые компоненты:**
- **QuantumNeuronB2** — квантоподобный нейрон, преобразующий фазовые углы через функции
- **WavePhaseLayer** — волновой слой, комбинирующий локальную и глобальную фазовую динамику
- **WaveStateMemory (WSM)** — память состояний с параметрами затухания и хаоса
- **global_phase_field** — вычисление глобального фазового поля для всей системы

### 2. **Память волновых состояний (Wave State Memory)**

Специализированный модуль для сохранения и эволюции фазовых состояний.

**Технология:**
- Отслеживание эволюции фазовых состояний во времени
- Параметры: gamma (основное затухание), delta (чувствительность), epsilon (глобальное взаимодействие)
- Режим REM (Rapid Eye Movement) — активация при высоком дрейфе параметров
- Детектирование хаотических режимов и добавление контролируемого шума
- Нормализация фазовых углов в диапазон 

**Параметры:**
- `gamma` — коэффициент сохранения предыдущего состояния (0-1)
- `delta` — чувствительность к локальным изменениям
- `epsilon` — влияние глобального фазового поля
- `chaos_amp` — амплитуда хаотических возмущений
- `rem_threshold` — порог активации режима REM

### 3. **Фазовая токенизация (Phase Tokenizer)**

Модуль `phase_tokenizer.py` для преобразования различных типов данных в фазовое представление.

**Технология:**
- Преобразование текста в фазовые векторы через анализ кодов символов
- FFT (быстрое преобразование Фурье) для спектрального анализа текстовых данных
- Адаптивное кодирование коротких сообщений через позиционное проецирование
- Обработка изображений через 2D FFT с фазовым извлечением
- Универсальная размерность представления (параметр `dim`)

**Процесс:**
- Текст → коды ASCII → спектральный анализ → фазовые углы
- Изображение → 2D спектр → фазовое извлечение → нормализация

### 4. **Битовая сетка и сенсор (BitGrid Sensor)**

Компонент `bitgrid.py` для кодирования изображений в битовые сетки с фазовым анализом.

**Технология:**
- Преобразование изображений в двоичные сетки фиксированного размера (по умолчанию 16×16)
- Спектральная фильтрация и реконструкция через фазовое пространство
- Вычисление пространственных моментов (центр масс, ориентация)
- Вычисление гистограмм пиксельных значений для статистического анализа
- Расстояние Вассерштейна для сравнения распределений
- Пересечение над объединением (IoU) для оценки сходства

**Особенности:**
- Адаптивное пороговое значение для бинаризации
- Морфологические операции (закрытие, открытие) для очистки шума
- Опциональное использование обученного декодера для восстановления битов
- Шаблонное хранилище (TemplateStore) для сохранения и поиска масок

### 5. **Гибридное обучение (Hybrid Training)**

Инновационный подход к обучению в `neural_network.py`.

**Технология:**
- **Фаза Backpropagation** — традиционное обратное распространение ошибки (первые эпохи)
- **Фаза фазовой динамики** — переключение на фазовые вычисления при достижении порога качества
- **Сохранение базовой конфигурации** — отслеживание дрейфа параметров от инициализации
- **Адаптивные амплитуды** — контролирование масштаба обновлений параметров
- **Температурное отжигание** — постепенное снижение уровня шума и возбуждения

**Процесс переключения:**
1. Инициализация с малыми весами
2. Обучение через backprop до эпохи ~10-15
3. Мониторинг потерь и точности
4. При потерях < 3.5 — переключение на фазовый режим
5. Сохранение базовой конфигурации для дальнейшего отслеживания

### 6. **Архетипические прототипы**

Компонент для эмоциональной категоризации.

**Технология:**
- Создание векторов-прототипов для каждого класса эмоций
- Инициализация через  фазовые смещения
- Постепенное уточнение прототипов во время обучения
- Использование для классификации через сравнение сходства

### 7. **Пиксель-архив (Iсos Pixyh Archive)**

Специальная структура данных для сохранения фазовых состояний.

**Технология:**
- Двумерная решётка хранилища для кодирования координат (x, y) в дискретные ячейки
- Периодическое вычисление волновой структуры в дискретной сетке
- Сохранение и восстановление фазовых состояний с фиксированными координатами
- Использование для создания фазовой памяти системы

### 8. **Эмпатичный ответчик датасета**

Компонент `EmpathicDatasetResponder` для работы с текстовыми диалогами.

**Технология:**
- Парсинг различных форматов датасетов (многокруговые диалоги, структурированные беседы)
- Извлечение контекста и ответов из сложных структур данных
- Кодирование ответов в фазовое пространство
- Поиск наиболее похожего ответа через косинусное сходство в фазовом пространстве
- Интеграция с метаинформацией об эмоциях

---

## 📊 Метрики и оценка качества

### Основные метрики

- **ArchSim** (Archetype Similarity) — косинусное сходство между фазовым состоянием и архетипом эмоции
- **RCL** (Resonant Clustering) — стандартное отклонение среднего фазового состояния, измеряет "резонансность"
- **Phase Distance** — различие между текущим фазовым состоянием и памятью системы
- **Drift** — мера отклонения параметров от базовой инициализации
- **Consciousness Score** — комбинированная метрика качества работы системы

### Компоненты оценки

- Точность классификации (accuracy)
- Стабильность фазовых состояний
- Выравнивание с архетипическими прототипами
- Кластеризация фазовых векторов (K-means)
- Анализ гистограмм и распределений

---

## 📁 Структура проекта

```
architector/
├── neural_network.py              # Основной модуль нейронной сети и обучения
├── wave_network_integrated.py     # Волновая архитектура и фазовые слои
├── phase_tokenizer.py             # Преобразование данных в фазовое пространство
├── bitgrid.py                     # Битовая сетка и сенсор изображений
├── quantum_neuron.py              # Квантовый симулятор и коррекция волн
├── inference.py                   # Модуль инференса
├── plotting.py                    # Визуализация результатов
├── functions.py                   # Вспомогательные функции (SELU и т.д.)
├── graph.py                       # Построение графиков динамики
├── phase_checks.py                # Диагностика фазовых состояний
├── prepare_cornell_dataset.py     # Подготовка датасета диалогов
├── train_bitgrid_decoder.py       # Обучение декодера битовых сеток
├── wavetensor_quantum_toy.py      # Экспериментальный модуль
├── datasets/                      # Директория датасетов
│   ├── cornell_movie_dialogs/     # Корнеллский датасет диалогов
│   └── movie_*.txt                # Исходные текстовые данные
├── eval_report/                   # Отчёты об оценке и метрики
│   ├── final_metrics.json         # Финальные метрики
│   ├── predictions.csv            # Предсказания модели
│   ├── confusion_matrix.png       # Матрица ошибок
│   └── *.png                      # Графики и визуализации
├── pokrov_model.pt                # Сохранённая обученная модель
├── logs_epoch_phase_metrics.pt    # Логи фазовых метрик по эпохам
├── bitgrid_decoder.pt             # Обученный декодер битовых сеток
├── foresight_matrix.npy           # Матрица предвидения (квантовый модуль)
├── requirements.txt               # Зависимости проекта
└── README.md                      # Этот файл
```

---

## 🔧 Технологический стек

### Основные библиотеки

- **PyTorch 2.2+** — основной фреймворк для глубокого обучения
- **NumPy 2.2+** — численные вычисления и работа с массивами
- **Transformers 4.37+** — предобученные модели и утилиты
- **Scikit-learn 1.6+** — машинное обучение и предварительная обработка
- **Matplotlib 3.10+** и **Seaborn 0.13+** — визуализация
- **SciPy 1.15+** — научные вычисления (FFT, Линдблад, матрицы плотности)
- **NetworkX 3.4+** — анализ графов
- **Joblib 1.3+** — параллелизм и сохранение объектов
- **Pillow 11.0+** — обработка изображений

### Опциональные зависимости

- **CUDA** — ускорение GPU вычислений
- **Dash 2.14+** — интерактивные веб-приложения

---

## 🚀 Установка и запуск

### Требования

- Python 3.13+
- pip или conda

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/crie123/architect
cd architect

# Установка зависимостей
pip install -r requirements.txt

# Для поддержки GPU (опционально)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Использование

```bash
# 1. Обучение модели
python neural_network.py

# 2. Инференс (предсказания)
python inference.py

# 3. Оценка метрик
python tests_and_evals/evaluate_phase_metrics.py

# 4. Оценка волновой модели
python tests_and_evals/evaluate_wave_model.py

# 5. Подготовка датасета
python prepare_cornell_dataset.py

# 6. Обучение битовой сетки
python train_bitgrid_decoder.py
```

---

## 🔬 Научные основы

### Волновая динамика

Система моделирует эволюцию фазовых полей, похожую на волновые процессы в физике:
- Локальные осцилляции (фазовые углы каждого нейрона)
- Глобальные интерференционные эффекты (взаимодействие всех нейронов)
- Дампинг и диссипация энергии (затухание через параметр gamma)
- Нелинейные эффекты через  трансформации

### Квантовая природа

Архитектура вдохновлена квантовой механикой:
- Фазовые углы как аналог квантовых фаз
- Матрицы плотности для описания смешанных состояний
- Уравнение Линдблада для декогеренции
- Операторы Паули и их собственные значения

### Эмоциональное моделирование

Система интерпретирует фазовые состояния как эмоциональные архетипы:
- Каждая эмоция — уникальный паттерн в фазовом пространстве
- Резонанс с прототипами — мера эмоциональной принадлежности
- Дрейф параметров — изменение эмоционального состояния со временем

---

## 📈 Расширенные возможности

### Режимы работы

1. **Обучение** — адаптация весов и параметров на размеченных данных
2. **Инференс** — предсказание эмоций на новых текстах
3. **Анализ** — детальное изучение фазовых состояний и метрик

### Мультимодальность

- **Текст** — через фазовую токенизацию
- **Изображения** — через битовую сетку и спектральный анализ
- **Диалоги** — с контекстом и историей взаимодействия

### Визуализация

- Тепловые карты фазовых состояний
- 3D визуализация фазового пространства
- Матрицы сознания и резонансные кластеры
- Гистограммы распределений
- Поля резонанса

---

## 📝 Лицензия

Защищено лицензией ARL-1 — Несанкционированное использование строго запрещено.

**Контакт:** Cyan11777@gmail.com

---

## 👤 Автор

© 2025 Nikitenko "crie123" Kirill. Все права защищены.

---

<a name="english"></a>

# Architector 🧠

[Return to Russian version](#русская-версия)

---

## 🇬🇧 English Version

### Project Description

**Architector** is an experimental machine learning system that models emotional intelligence and personality dynamics through wave-based phase states of a neural network. The project combines classical deep learning methods with innovative approaches using quantum-like computations, wave dynamics, and phase architecture for processing emotional data and natural language.

### Key Features

- 🌊 **Wave neural network** with multilevel phase dynamics
- 🎭 **Emotion processing** — classification and analysis of emotional states from text
- 🧪 **Hybrid architecture** — combining traditional backpropagation with phase computations
- 🔬 **Quantum-like elements** — simulating quantum phenomena in classical neural networks
- 💾 **Dynamic model saving** — archiving phase states
- 📊 **Advanced visualization** — phase maps, consciousness matrices, 3D visualizations
- 🖼️ **Multimodal processing** — text, images, and bit grids
- 🔄 **Training and inference modes** — flexible configurations for various scenarios

---

## 🏗️ Architectural Components and Technologies

### 1. **Wave Neural Network**

The core component of the project, located in `wave_network_integrated.py`.

**Technology:**
- Data processing through phase space instead of traditional vector representation
- Each neuron operates on phase angles representing system state
- Multilayer architecture with a chain of wave layers
- Synthesis of local and global phase fields for complex processing

**Key Components:**
- **QuantumNeuronB2** — quantum-like neuron transforming phase angles through  functions
- **WavePhaseLayer** — wave layer combining local and global phase dynamics
- **WaveStateMemory (WSM)** — state memory with damping and chaos parameters
- **global_phase_field** — computing global phase field for the entire system

### 2. **Wave State Memory (WSM)**

Specialized module for preserving and evolving phase states.

**Technology:**
- Tracking evolution of phase states over time
- Parameters: gamma (main damping), delta (sensitivity), epsilon (global interaction)
- REM mode (Rapid Eye Movement) — activation under high parameter drift
- Detection of chaotic regimes and controlled noise addition
- Phase angle normalization to range 

**Parameters:**
- `gamma` — coefficient for preserving previous state (0-1)
- `delta` — sensitivity to local changes
- `epsilon` — influence of global phase field
- `chaos_amp` — amplitude of chaotic perturbations
- `rem_threshold` — activation threshold for REM mode

### 3. **Phase Tokenization**

Module `phase_tokenizer.py` for converting various data types into phase representation.

**Technology:**
- Converting text to phase vectors through character code analysis
- FFT (Fast Fourier Transform) for spectral analysis of textual data
- Adaptive encoding of short messages through positional projection
- Image processing through 2D FFT with phase extraction
- Universal representation dimensionality (parameter `dim`)

**Process:**
- Text → ASCII codes → spectral analysis → phase angles
- Image → 2D spectrum → phase extraction → normalization

### 4. **Bit Grid and Sensor**

Component `bitgrid.py` for encoding images into bit grids with phase analysis.

**Technology:**
- Converting images into binary grids of fixed size (default 16×16)
- Spectral filtering and reconstruction through phase space
- Computing spatial moments (center of mass, orientation)
- Computing pixel value histograms for statistical analysis
- Wasserstein distance for comparing distributions
- Intersection over Union (IoU) for similarity assessment

**Features:**
- Adaptive thresholding for binarization
- Morphological operations (closing, opening) for noise cleanup
- Optional use of trained decoder for bit recovery
- Template storage (TemplateStore) for mask saving and retrieval

### 5. **Hybrid Training**

Innovative training approach in `neural_network.py`.

**Technology:**
- **Backpropagation phase** — traditional error backpropagation (early epochs)
- **Phase dynamics phase** — switching to phase computations when quality threshold reached
- **Baseline preservation** — tracking parameter drift from initialization
- **Adaptive amplitudes** — controlling scale of parameter updates
- **Temperature annealing** — gradual reduction of noise and excitation levels

**Switching Process:**
1. Initialization with small weights
2. Training via backprop until epoch ~10-15
3. Monitoring loss and accuracy
4. When loss < 3.5 — switching to phase mode
5. Saving baseline configuration for further tracking

### 6. **Archetypal Prototypes**

Component for emotion categorization.

**Technology:**
- Creating prototype vectors for each emotion class
- Initialization through  phase offsets
- Gradual refinement of prototypes during training
- Usage for classification through similarity comparison

### 7. **Pixel Archive (Icos Pixyh Archive)**

Special data structure for preserving phase states.

**Technology:**
- Two-dimensional storage grid encoding coordinates (x, y) into discrete cells
- Periodic computation of wave structure in discrete grid
- Saving and retrieving phase states with fixed coordinates
- Usage for creating system phase memory

### 8. **Empathic Dataset Responder**

Component `EmpathicDatasetResponder` for working with text dialogues.

**Technology:**
- Parsing various dataset formats (multi-turn dialogues, structured conversations)
- Extracting context and responses from complex data structures
- Encoding responses into phase space
- Finding most similar response through cosine similarity in phase space
- Integration with emotion metadata

---

## 📊 Metrics and Quality Assessment

### Primary Metrics

- **ArchSim** (Archetype Similarity) — cosine similarity between phase state and emotion archetype
- **RCL** (Resonant Clustering) — standard deviation of mean phase state, measuring "resonance"
- **Phase Distance** — difference between current phase state and system memory
- **Drift** — measure of parameter deviation from baseline initialization
- **Consciousness Score** — combined system quality metric

### Assessment Components

- Classification accuracy
- Phase state stability
- Alignment with archetypal prototypes
- Phase vector clustering (K-means)
- Histogram and distribution analysis

---

## 📁 Project Structure

```
architector/
├── neural_network.py              # Main neural network and training module
├── wave_network_integrated.py     # Wave architecture and phase layers
├── phase_tokenizer.py             # Converting data to phase space
├── bitgrid.py                     # Bit grid and image sensor
├── quantum_neuron.py              # Quantum simulator and wave correction
├── inference.py                   # Inference module
├── plotting.py                    # Results visualization
├── functions.py                   # Helper functions (SELU, etc.)
├── graph.py                       # Dynamics graph plotting
├── phase_checks.py                # Phase state diagnostics
├── prepare_cornell_dataset.py     # Dialog dataset preparation
├── train_bitgrid_decoder.py       # Bit grid decoder training
├── wavetensor_quantum_toy.py      # Experimental module
├── datasets/                      # Datasets directory
│   ├── cornell_movie_dialogs/     # Cornell dialog dataset
│   └── movie_*.txt                # Raw text data
├── eval_report/                   # Evaluation reports and metrics
│   ├── final_metrics.json         # Final metrics
│   ├── predictions.csv            # Model predictions
│   ├── confusion_matrix.png       # Error matrix
│   └── *.png                      # Graphs and visualizations
├── pokrov_model.pt                # Saved trained model
├── logs_epoch_phase_metrics.pt    # Phase metrics logs by epoch
├── bitgrid_decoder.pt             # Trained bit grid decoder
├── foresight_matrix.npy           # Foresight matrix (quantum module)
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

---

## 🔧 Technology Stack

### Core Libraries

- **PyTorch 2.2+** — main deep learning framework
- **NumPy 2.2+** — numerical computations and array operations
- **Transformers 4.37+** — pretrained models and utilities
- **Scikit-learn 1.6+** — machine learning and preprocessing
- **Matplotlib 3.10+** and **Seaborn 0.13+** — visualization
- **SciPy 1.15+** — scientific computing (FFT, Lindblad, density matrices)
- **NetworkX 3.4+** — graph analysis
- **Joblib 1.3+** — parallelization and object serialization
- **Pillow 11.0+** — image processing

### Optional Dependencies

- **CUDA** — GPU acceleration for computations
- **Dash 2.14+** — interactive web applications

---

## 🚀 Installation and Usage

### Requirements

- Python 3.13+
- pip or conda

### Installation

```bash
# Clone repository
git clone https://github.com/crie123/architect
cd architect

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Usage

```bash
# 1. Train model
python neural_network.py

# 2. Inference (predictions)
python inference.py

# 3. Evaluate metrics
python tests_and_evals/evaluate_phase_metrics.py

# 4. Evaluate wave model
python tests_and_evals/evaluate_wave_model.py

# 5. Prepare dataset
python prepare_cornell_dataset.py

# 6. Train bit grid decoder
python train_bitgrid_decoder.py
```

---

## 🔬 Scientific Foundation

### Wave Dynamics

The system models evolution of phase fields similar to wave processes in physics:
- Local oscillations (phase angles of each neuron)
- Global interference effects (interaction of all neurons)
- Energy damping and dissipation (decay through gamma parameter)
- Nonlinear effects through  transformations

### Quantum Nature

Architecture inspired by quantum mechanics:
- Phase angles as analogues of quantum phases
- Density matrices for describing mixed states
- Lindblad equation for decoherence
- Pauli operators and their eigenvalues

### Emotional Modeling

System interprets phase states as emotional archetypes:
- Each emotion — unique pattern in phase space
- Resonance with prototypes — measure of emotional belonging
- Parameter drift — emotional state change over time

---

## 📈 Advanced Capabilities

### Operating Modes

1. **Training** — adapting weights and parameters on labeled data
2. **Inference** — predicting emotions on new texts
3. **Analysis** — detailed study of phase states and metrics

### Multimodality

- **Text** — through phase tokenization
- **Images** — through bit grid and spectral analysis
- **Dialogues** — with context and interaction history

### Visualization

- Phase state heatmaps
- 3D phase space visualization
- Consciousness matrices and resonant clusters
- Distribution histograms
- Resonance fields

---

## 📝 License

Protected by ARL-1 license — Unauthorized use strictly prohibited.

**Contact:** Cyan11777@gmail.com

---

## 👤 Author

© 2025 Nikitenko "crie123" Kirill. All rights reserved.

---

**Последнее обновление / Last Updated:** November 27, 2025
