from collections import Counter


class Tokenizer:
    """Tokenizer class for converting text into tokens and building a vocabulary."""

    def __init__(
        self,
        min_freq: int = 1,
        max_vocab: int = 10000,
        max_sequence_length: int = 64,
    ):
        """Initialize the Tokenizer.

        Args:
            min_freq (int): Minimum frequency for a token to be included in the vocabulary. Default is 1.
            max_vocab (int): Maximum size of the vocabulary. Default is 10000.
            max_sequence_length (int): Maximum length of the tokenized sequences. Default is 64.
        """
        self.vocabulary: dict[str, int] = {}
        self.inverse_vocabulary: dict[int, str] = {}
        self.min_frequency = min_freq
        self.max_vocabulary = max_vocab
        self.max_sequence_length = max_sequence_length

        self.padding_token = "[PAD]"
        self.masking_token = "[MASK]"
        self.unknown_token = "[UNK]"
        self.start_token = "[SOS]"
        self.end_token = "[EOS]"

    def tokenize(self, text: str) -> list[str]:
        """Tokenize the input text into a list of tokens.

        Args:
            text: A string representing the input text.

        Returns:
            A list of tokens.
        """
        # Convert text to lowercase
        text = text.lower()
        # Remove punctuation
        for punct in [",", ".", "?", "!", ";", ":", "(", ")", '"', "'"]:
            text = text.replace(punct, "")
        # Split by whitespace and remove empty tokens
        return [token for token in text.strip().split() if token]

    def build_vocabulary(self, texts: list[str]):
        """Build the vocabulary from a list of texts.

        Args:
            texts: A list of strings representing the input texts.
        """
        token_counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            token_counter.update(tokens)

        # Filter tokens based on minimum frequency and sort by frequency and lexicographical order
        items = [t for t, f in token_counter.items() if f >= self.min_frequency]
        items = sorted(items, key=lambda x: (-token_counter[x], x))
        items = items[: self.max_vocabulary - 5]

        # Add special tokens to the vocabulary and build the vocabulary and inverse vocabulary
        idx = 0
        for token in [
            self.padding_token,
            self.masking_token,
            self.unknown_token,
            self.start_token,
            self.end_token,
        ] + items:
            self.vocabulary[token] = idx
            self.inverse_vocabulary[idx] = token
            idx += 1

    def load_vocabulary(self, vocab_file: str):
        """Load the vocabulary from a file as "token, index".

        Args:
            vocab_file: Path to the vocabulary file.
        """
        with open(vocab_file, "r") as f:
            self.vocabulary = {
                line.strip().split(",")[0]: int(line.strip().split(",")[1])
                for line in f
            }
            self.inverse_vocabulary = {
                idx: token for token, idx in self.vocabulary.items()
            }

    def save_vocabulary(self, vocab_file: str):
        """Save the vocabulary to a file as "token, index".

        Args:
            vocab_file: Path to the vocabulary file.
        """
        with open(vocab_file, "w") as f:
            for token in self.vocabulary:
                f.write(f"{token}, {self.vocabulary[token]}\n")

    def encode(self, text: str) -> list[int]:
        """Encode the input text into a list of token indices.

        Args:
            text: A string representing the input text.

        Returns:
            A list of token indices.
        """
        raw_tokens = self.tokenize(text)
        tokens = [self.start_token] + raw_tokens + [self.end_token]

        if len(tokens) > self.max_sequence_length:
            tokens = tokens[: self.max_sequence_length - 1] + [self.end_token]

        padding_length = self.max_sequence_length - len(tokens)
        tokens += [self.padding_token] * padding_length

        return [
            self.vocabulary.get(token, self.vocabulary[self.unknown_token])
            for token in tokens
        ]

    def decode(self, token_indices: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token indices back into a string.

        Args:
            token_indices: A list of token indices.
            skip_special_tokens: Whether to skip special tokens in the output. Default is True.

        Returns:
            A string representing the decoded text.
        """
        special_tokens = {
            self.vocabulary.get(self.padding_token),
            self.vocabulary.get(self.start_token),
            self.vocabulary.get(self.end_token),
            self.vocabulary.get(self.masking_token),
        }

        tokens = []
        for idx in token_indices:
            if skip_special_tokens and idx in special_tokens:
                continue
            token = self.inverse_vocabulary.get(idx, self.unknown_token)
            tokens.append(token)

        return " ".join(tokens)

    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary.

        Returns:
            The size of the vocabulary.
        """
        return len(self.vocabulary)
