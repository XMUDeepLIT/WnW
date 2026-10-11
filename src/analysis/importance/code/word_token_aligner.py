
import re


class WordTokenAligner:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _tokenize_with_positions(self, text: str) -> list[dict]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens_with_positions = []
        accumulated = ""
        for tok_id in token_ids:
            tok_text = self.tokenizer.decode([tok_id])
            char_start = len(accumulated)
            accumulated += tok_text
            char_end = len(accumulated)
            tokens_with_positions.append({
                "token_id": tok_id,
                "token_text": tok_text,
                "char_start": char_start,
                "char_end": char_end,
            })
        assert accumulated == text, f"Tokenization mismatch: '{accumulated}' != '{text}'"
        return tokens_with_positions

    def align_words_to_tokens(
        self,
        text: str,
        word_segments: list[dict],
        min_word_score: float = 0.0,
    ) -> list[dict]:
        """Return per-token dicts with the parent word's timestamps.

        word_segments: list of {word, start, end, score?} from WhisperX.
        Tokens with no overlapping word (e.g. standalone punctuation) are skipped.
        """
        tokens = self._tokenize_with_positions(text)

        filtered_words = [ws for ws in word_segments if ws.get("score", 0.0) >= min_word_score]

        word_positions = []
        current_pos = 0
        for word_seg in filtered_words:
            word = word_seg["word"]
            found_pos = text.find(word, current_pos)
            if found_pos == -1:
                word_clean = re.sub(r'[^\w\s]', '', word)
                if word_clean:
                    found_pos = text.find(word_clean, current_pos)
            if found_pos == -1:
                print(f"Warning: Cannot find word '{word}' in text at position {current_pos}")
                continue
            word_positions.append({
                "word": word,
                "char_start": found_pos,
                "char_end": found_pos + len(word),
                "start": word_seg["start"],
                "end": word_seg["end"],
                "score": word_seg.get("score", 0.0),
            })
            current_pos = found_pos + len(word)

        token_word_mapping = []
        word_idx = 0
        for tok_idx, tok in enumerate(tokens):
            best_word_idx = None
            best_overlap = 0
            for w_idx in range(max(0, word_idx - 2), min(len(word_positions), word_idx + 3)):
                wp = word_positions[w_idx]
                overlap_start = max(tok["char_start"], wp["char_start"])
                overlap_end = min(tok["char_end"], wp["char_end"])
                overlap = max(0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_word_idx = w_idx
            if best_word_idx is not None and best_overlap > 0:
                wp = word_positions[best_word_idx]
                token_word_mapping.append({
                    "token_id": tok["token_id"],
                    "token_text": tok["token_text"],
                    "word": wp["word"],
                    "word_start": wp["start"],
                    "word_end": wp["end"],
                    "word_score": wp["score"],
                    "token_idx": tok_idx,
                })
                word_idx = best_word_idx

        return token_word_mapping
