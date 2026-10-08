from collections import Counter


class Tokenizer:
    """Tokenizer class for converting text into tokens and building a vocabulary."""

    def __init__(self, min_freq: int = 1, max_vocab: int = 10000):
        """Initialize the Tokenizer.

        Args:
            min_freq: Minimum frequency for a token to be included in the vocabulary. Default is 1.
            max_vocab: Maximum size of the vocabulary. Default is 10000.
        """
        self.vocabulary: dict[str, int] = {}
        self.inverse_vocabulary: dict[int, str] = {}
        self.min_frequency = min_freq
        self.max_vocabulary = max_vocab
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

    # TODO: vocabulary can be loaded from a file or pre-defined list in JSON format
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
        tokens = self.tokenize(text)
        return [
            self.vocabulary.get(token, self.vocabulary[self.unknown_token])
            for token in tokens
        ]

    def decode(self, token_indices: list[int]) -> str:
        """Decode a list of token indices back into a string.

        Args:
            token_indices: A list of token indices.

        Returns:
            A string representing the decoded text.
        """
        tokens = [
            self.inverse_vocabulary.get(idx, self.unknown_token)
            for idx in token_indices
        ]
        return " ".join(tokens)
