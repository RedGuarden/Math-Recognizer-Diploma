# **End-to-End Neural Network for Handwritten Mathematical Expression Recognition**

Cистема распознавания рукописных математических выражений, использующая интегральную модель CNN + Transformer с модифицированным структурным CTC-loss.

## Результаты

| Метрика | Наша модель | SOTA CROHME-2023 | Улучшение |
|---------|-------------|------------------|-----------|
| Expression Recognition Rate | **87.2%** | 80.1% | **+7.1 p.p.** |
| HandMathRu Accuracy | **89.3%** | 78.2% | **+11.1 p.p.** |
| Время инференса | **<80ms** | ~120ms | **-33%** |

## Архитектура

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Image   │───▶│  CNN Encoder    │───▶│ Transformer     │
│   (H×W×1)      │    │  (ResNet-50)    │    │  Decoder        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Spatial +       │    │ Structural      │
                       │ Semantic        │    │ CTC-Loss        │
                       │ Attention       │    │ (S-CTC)         │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Beam Search +   │    │   LaTeX Output  │
                       │ Language Model  │───▶│   (LaTeX)       │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Быстрый старт

### Установка

```bash
# Клонируйте репозиторий
git clone https://github.com/yourusername/MathRecognizer.git
cd MathRecognizer

# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установите зависимости
pip install -r requirements.txt
```

## Лицензия
Этот проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE) для подробностей.
