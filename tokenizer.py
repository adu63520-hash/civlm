"""
字节级 BPE 分词器（改自 energy_llm_v2/tokenizer.py）
- 数字按单个数位切分：118 → 1 1 8，模型才能学到数值之间的关系
- 词表 2048：文明史文本的词汇很有限，小词表让嵌入层更省参数
"""
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|endoftext|>"]


def train_bpe(text_iterator, vocab_size, save_path):
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tk.train_from_iterator(text_iterator, trainer)
    tk.save(save_path)
    return BPETokenizer(save_path)


class BPETokenizer:
    def __init__(self, path):
        self.tk = Tokenizer.from_file(path)
        self.eot_id = self.tk.token_to_id("<|endoftext|>")

    @property
    def vocab_size(self):
        return self.tk.get_vocab_size()

    def encode(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts):
        return [e.ids for e in self.tk.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids):
        return self.tk.decode(list(ids), skip_special_tokens=True)
