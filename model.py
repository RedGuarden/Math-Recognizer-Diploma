import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_V2_S_Weights
import json
import math

# Классы Tokenizer и PositionalEncoding остаются без изменений
class Tokenizer:
    """
    Обрабатывает токенизацию и детокенизацию с использованием готового словаря.
    """
    def __init__(self, vocab_path):
        """
        Аргументы:
            vocab_path (str): Путь к файлу vocab.json.
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        
        self.pad_id = self.token_to_id['<pad>']
        self.sos_id = self.token_to_id['<sos>']
        self.eos_id = self.token_to_id['<eos>']
        self.unk_id = self.token_to_id['<unk>']
        
    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def encode(self, latex_tokens):
        """Преобразует список токенов LaTeX в список ID токенов."""
        return [self.token_to_id.get(token, self.unk_id) for token in latex_tokens]

    def decode(self, token_ids, strip_special_tokens=True):
        """Преобразует список ID токенов обратно в строку LaTeX."""
        tokens = []
        for token_id in token_ids:
            if strip_special_tokens and token_id in [self.sos_id, self.eos_id, self.pad_id]:
                continue
            tokens.append(self.id_to_token.get(token_id, ''))
        return "".join(tokens)

class PositionalEncoding(nn.Module):
    """
    Добавляет позиционную информацию во входные эмбеддинги.
    Из оригинальной статьи о Трансформерах.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EfficientNetEncoder(nn.Module):
    """
    Кодировщик на основе CNN, использующий предварительно обученную модель EfficientNetV2-S.
    """
    def __init__(self, encoded_image_size=7): # EfficientNet по умолчанию выводит карту признаков размером 7x7
        super(EfficientNetEncoder, self).__init__()
        # Используем EfficientNetV2-S с последними рекомендованными весами
        effnet = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        
        # Нам нужна только часть, извлекающая признаки
        self.features = effnet.features
        
        # Выходная карта признаков будет иметь глубину 1280 для EfficientNetV2-S
        self.encoder_output_dim = 1280
        
        # Адаптивный пулинг для обеспечения фиксированного размера вывода, хотя EffNet это уже делает.
        # Это делает модель устойчивой к разным размерам входных изображений, если потребуется.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        # images: (batch_size, 1, H, W) -> нам нужно 3 канала для EfficientNet
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)

        features = self.features(images) # (batch_size, 1280, H/32, W/32) -> для EffNet это (batch_size, 1280, 7, 7) для входа 224x224
        features = self.adaptive_pool(features) # (batch_size, 1280, encoded_image_size, encoded_image_size)
        features = features.permute(0, 2, 3, 1) # (batch_size, size, size, 1280)
        
        # "Выравниваем" пространственные измерения
        batch_size = features.size(0)
        features = features.view(batch_size, -1, self.encoder_output_dim) # (batch_size, size*size, 1280)
        
        return features

# Класс TransformerDecoder остается без изменений
class TransformerDecoder(nn.Module):
    """
    Декодер на основе Трансформера для генерации последовательности LaTeX.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, memory, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Аргументы:
            memory (Tensor): Последовательность с последнего слоя кодировщика. (batch_size, seq_len_encoder, d_model)
            tgt (Tensor): Последовательность для декодера. (batch_size, seq_len_decoder)
            tgt_mask (Tensor): Квадратная маска внимания для целевой последовательности.
            tgt_key_padding_mask (Tensor): Маска для токенов заполнения (padding) в целевой последовательности.
        """
        # Встраиваем целевую последовательность и добавляем позиционное кодирование
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        # Примечание: оригинальное позиционное кодирование может потребовать корректировки, если используется batch_first=True
        # Изменим форму для pos_encoder, а затем вернем обратно
        # Текущий pos_encoder ожидает (seq_len, batch_size, d_model)
        # Наш tgt_emb имеет форму (batch_size, seq_len, d_model)
        tgt_emb = tgt_emb.permute(1, 0, 2)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = tgt_emb.permute(1, 0, 2)

        # Декодируем
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        
        # Генерируем предсказания
        return self.generator(output)

# --- НОВАЯ МОДЕЛЬ V2 ---
class Image2LatexModelV2(nn.Module):
    """
    Основная модель, объединяющая НОВЫЙ кодировщик и декодер.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(Image2LatexModelV2, self).__init__()
        
        self.encoder = EfficientNetEncoder()
        
        # Линейный слой для проекции вывода кодировщика в ожидаемую размерность декодера
        self.encoder_projection = nn.Linear(self.encoder.encoder_output_dim, d_model)
        
        self.decoder = TransformerDecoder(vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, dropout)
        
    def generate_square_subsequent_mask(self, sz, device):
        # Генерирует булеву маску для причинного (последовательного) внимания.
        mask = torch.triu(torch.full((sz, sz), True, device=device), diagonal=1)
        return mask

    def forward(self, src_img, tgt_seq, tgt_padding_mask=None):
        """
        Аргументы:
            src_img (Tensor): Входной тензор изображения. (batch_size, channels, H, W)
            tgt_seq (Tensor): Целевая последовательность для teacher forcing. (batch_size, seq_len)
            tgt_padding_mask (Tensor): Маска для токенов заполнения в целевой последовательности.
        """
        # Кодируем изображение
        encoded_img = self.encoder(src_img)
        memory = self.encoder_projection(encoded_img)
        
        # Создаем причинную маску для декодера
        tgt_seq_len = tgt_seq.size(1)
        device = tgt_seq.device
        
        # Поскольку слой декодера использует batch_first, маска должна быть (N, L, S) или (L,S)
        # Сгенерируем маску размером (tgt_seq_len, tgt_seq_len)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        
        # Получаем предсказания от декодера
        predictions = self.decoder(memory, tgt_seq, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        
        return predictions 