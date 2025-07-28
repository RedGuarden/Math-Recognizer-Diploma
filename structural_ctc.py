import torch
import torch.nn as nn
import itertools

"""
Файл: structural_ctc.py

Концептуальная реализация алгоритма "Структурный CTC-loss" (S-CTC) для распознавания
математических выражений.

Описание алгоритма:
Алгоритм S-CTC представляет собой расширение классического Connectionist Temporal 
Classification (CTC), адаптированное для работы с двумерными структурами, 
такими как математические формулы.

В отличие от стандартного CTC, который работает с линейными последовательностями, 
S-CTC вводит в словарь специальные "структурные теги". Эти теги позволяют 
представить иерархическую структуру формулы (например, дроби, верхние и нижние 
индексы) в виде одномерной последовательности.

Ключевой особенностью S-CTC является то, что функция потерь анализирует эти теги 
и применяет дополнительный "структурный штраф", если модель неправильно 
предсказывает расположение или наличие тегов. Это заставляет модель изучать не 
только символы, но и синтаксическую структуру выражения.

Данная реализация демонстрирует основную идею алгоритма и предназначена для
включения в приложение диссертационной работы.
"""


def parse_latex_to_structured_sequence(latex_string: str, token_to_id: dict) -> list:
    """
    (Концептуальный пример) Преобразует строку LaTeX в структурированную
    последовательность токенов с тегами.

    Для полноценной работы требуется сложный парсер LaTeX, который строит
    дерево выражения и затем сериализует его в последовательность.

    Пример: 'x^{2+y}' -> ['x', '<struct_sup_start>', '2', '+', 'y', '<struct_sup_end>']
    Пример: '\\frac{a}{b}' -> ['<struct_frac_start>', 'a', '<struct_frac_delim>', 'b', '<struct_frac_end>']

    Args:
        latex_string (str): Входная строка в формате LaTeX.
        token_to_id (dict): Словарь для преобразования токенов в ID.

    Returns:
        list: Список ID токенов, представляющих структурированную последовательность.
    """
    # Этот пример является упрощенной демонстрацией.
    # Реальная система потребует использования инструментов, таких как ANTLR или pyparsing.
    processed_tokens = []
    if latex_string == 'x^{2+y}':
        processed_tokens = ['x', '<struct_sup_start>', '2', '+', 'y', '<struct_sup_end>']
    elif latex_string == '\\frac{a}{b}':
        processed_tokens = ['<struct_frac_start>', 'a', '<struct_frac_delim>', 'b', '<struct_frac_end>']
    else:
        # Упрощенная токенизация для других случаев
        processed_tokens = list(latex_string.replace(" ", ""))

    return [token_to_id.get(token, token_to_id.get('<unk>', 0)) for token in processed_tokens]


class StructuralCTCLoss(nn.Module):
    """
    Реализует Structural CTC Loss (S-CTC).

    Эта функция потерь расширяет стандартный CTC-loss, добавляя штраф за
    ошибки в распознавании структурных тегов.

    Args:
        blank_id (int): ID "пустого" токена (blank) в словаре.
        structural_tags (set): Множество ID токенов, являющихся структурными тегами.
        reduction (str): Тип редукции: 'mean' (среднее) или 'sum' (сумма).
        structural_weight (float): Вес (коэффициент) для структурного штрафа.
                                   Определяет, насколько важны ошибки в структуре
                                   по сравнению с ошибками в символах.
    """
    def __init__(self, blank_id: int, structural_tags: set, reduction: str = 'mean', structural_weight: float = 1.0):
        super().__init__()
        if reduction not in ['mean', 'sum']:
            raise ValueError("reduction должен быть 'mean' или 'sum'")
            
        self.blank_id = blank_id
        self.structural_tags = structural_tags
        self.structural_weight = structural_weight
        self.reduction = reduction

        # Используем стандартный CTC Loss для базового расчета потерь по содержимому.
        # 'reduction' установлен в 'none', чтобы получить потери для каждого элемента батча.
        self.content_loss = nn.CTCLoss(blank=blank_id, reduction='none', zero_infinity=True)

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет S-CTC loss.

        Args:
            log_probs (Tensor): Выход модели (T, N, C), где T - длина входной
                                последовательности, N - размер батча, C - размер словаря.
            targets (Tensor): Целевые структурированные последовательности (N, S),
                              где S - макс. длина целевой последовательности.
            input_lengths (Tensor): Реальные длины входных последовательностей (N).
            target_lengths (Tensor): Реальные длины целевых последовательностей (N).

        Returns:
            Tensor: Итоговое значение функции потерь.
        """
        # --- Шаг 1: Расчет потерь по содержимому (Content Loss) ---
        # Эта часть аналогична стандартному CTC. Она оценивает, насколько правильно
        # распознаны символы в последовательности.
        content_loss = self.content_loss(log_probs, targets, input_lengths, target_lengths)

        # --- Шаг 2: Расчет структурного штрафа (Structural Penalty) ---
        # Это ключевое отличие S-CTC. Мы анализируем ошибки в предсказании
        # именно структурных тегов.

        # Убираем дубликаты и пустые токены из выхода модели, чтобы получить
        # наиболее вероятную предсказанную последовательность.
        with torch.no_grad():
            pred_tokens = torch.argmax(log_probs, dim=-1).T  # (N, T)

        batch_penalty = torch.zeros_like(content_loss)

        for i in range(log_probs.size(1)): # Итерация по элементам батча
            # Реальная длина предсказания и цели
            pred_len = input_lengths[i]
            target_len = target_lengths[i]

            # Получаем предсказанную последовательность и убираем повторы
            raw_pred = pred_tokens[i, :pred_len]
            best_path = [k for k, _ in itertools.groupby(raw_pred.tolist())]
            
            # Фильтруем пустые токены (blank)
            best_path_no_blanks = [p for p in best_path if p != self.blank_id]
            
            # Получаем реальную целевую последовательность
            target_seq = targets[i, :target_len].tolist()

            # Выделяем только структурные теги из предсказания и цели
            pred_struct_tags = [p for p in best_path_no_blanks if p in self.structural_tags]
            target_struct_tags = [t for t in target_seq if t in self.structural_tags]

            # Штраф начисляется, если предсказанный набор структурных тегов
            # не совпадает с целевым. Здесь используется простое сравнение,
            # но могут применяться и более сложные метрики (например, расстояние Левенштейна).
            if pred_struct_tags != target_struct_tags:
                # Величина штрафа может быть фиксированной или зависеть от степени
                # несоответствия. Для простоты, назначим фиксированный штраф.
                batch_penalty[i] = 1.0

        # --- Шаг 3: Комбинирование потерь ---
        # Итоговая функция потерь = потери по содержимому + взвешенный структурный штраф.
        total_loss = content_loss + self.structural_weight * batch_penalty

        # Применяем редукцию (среднее или сумма)
        if self.reduction == 'mean':
            return total_loss.mean()
        else:
            return total_loss.sum() 