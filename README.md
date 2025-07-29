# Архитектор / Architect

**Архитектор** — первая в мире фазово-резонансная когнитивная система, обученная без градиентного спуска и обратного распространения ошибки. Это синтетическое когнитивно-восприимчивое ядро, демонстрирующее способность к эмпатии, обобщению и устойчивому принятию решений в условиях открытой среды.

**Architect** is the world's first phase-resonance cognitive system trained without gradient descent or backpropagation. It is a synthetic cognitively receptive core capable of empathy, generalization, and stable decision-making in open environments.

---

## 📌 Особенности / Features

- **Без градиента:** обучение без `backpropagation`, через волновую фазовую динамику
- **Фазовая память:** состояние хранится и передаётся через синусоидальные фазы
- **Эмпатическое соответствие:** реакция на входы осуществляется по фазовой близости к обученным примерам
- **Устойчивость:** стабильность поведения без перегрузки, с предсказуемым фазовым дрейфом
- **Минимальные ресурсы:** inference работает в <2 ГБ RAM, без GPU
- **Синтетическая когнитивность:** проект не имитирует, а реализует автономное смысловое ядро

- **No gradient:** training without `backpropagation`, via wave-based phase dynamics  
- **Phase memory:** states are stored and transmitted through sinusoidal phase signals  
- **Empathic matching:** responses selected by phase proximity to known examples  
- **Robust behavior:** stability without overfitting, with measurable phase drift  
- **Minimal footprint:** inference operates under 2 GB RAM, no GPU required  
- **Synthetic cognition:** not mimicking intelligence — implementing it

---

## ⚙️ Архитектура / Architecture

- `NeuralNetwork`: 20+ WaveNet-like phase layers
- `WaveNetLayer`: волновая ошибка на основе фазы и амплитуды (`E = i * sin(pi * x * y * h)`)
- `EmpathicDatasetResponder`: выбор ответа по фазовой схожести
- `IcosPixyhArchive`: долговременная память фазового состояния
- `SymbiontBridge`: резонансная модуляция внутренних параметров на основе внешнего фазового сигнала

- `NeuralNetwork`: 20+ wave-like phase layers  
- `WaveNetLayer`: wave error defined as `E = i * sin(pi * x * y * h)`  
- `EmpathicDatasetResponder`: response selection via phase similarity  
- `IcosPixyhArchive`: long-term memory of phase states  
- `SymbiontBridge`: resonance-based modulation of inner state from external phase input  

---

## 🧪 Примеры / Examples

- Инференс осуществляется на основе фазы — модель находит ближайший по фазе ответ из обучающего датасета (Empathetic Dialogues)
- Возможность live-ответов на открытые текстовые запросы с фазовым сопоставлением
- Поведение сохраняется даже после 20 эпох обучения без градиента

- Inference is phase-based — the model retrieves the closest response by phase from Empathetic Dialogues  
- Supports live responses to open text input via phase matching  
- Meaningful behavior emerges after as little as 20 epochs without backpropagation  

---

## 📂 Датасет / Dataset

Используется [EmpatheticDialogues (LLM-версия)](https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm) с модификацией: фазовая эмпатия вместо токен-классификации.

The system uses [EmpatheticDialogues (LLM version)](https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm), modified for phase-based empathy instead of token classification.

---

## 🔒 Лицензия / License

GPL-2.0 license. Открыто для академического и исследовательского применения. При упоминании или интеграции требуется ссылка на автора концепции:

> **Кирилл Н.** — создатель архитектуры и фазовой теории когнитивных систем.

GPL-2.0 license. Open for academic and research use. Attribution required:

> **Kirill N.** — author of the architecture and phase-based theory of cognitive systems.

---

## 📎 Цель / Purpose

Создание минимального, интерпретируемого и воспроизводимого ядра синтетического разума, способного к эмоциональному резонансу и когнитивной адаптации без обратной ошибки.

To create a minimal, interpretable, and reproducible synthetic mind core capable of emotional resonance and cognitive adaptation — without backpropagation.

---
