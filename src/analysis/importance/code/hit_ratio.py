
import torch


class HitRatioCalculator:
    def __init__(self, top_k: int = 24, audio_only_topk: bool = False,
                 token_rate: float = 12.5):
        self.top_k = top_k
        self.audio_only_topk = audio_only_topk
        self.token_rate = token_rate

    def compute(
        self,
        attention_scores: torch.Tensor,
        num_audio: int,
        token_timestamps: list[dict],
    ) -> dict:
        """
        Args:
            attention_scores: [num_layers, num_heads, num_answer_tokens,
                               num_audio + num_answer]. Last dim layout:
                              [0, num_audio) = audio tokens,
                              [num_audio, ...] = answer tokens.
            num_audio: number of audio tokens.
            token_timestamps: list of {token_idx, word_start, word_end}.

        Returns dict:
            hit_ratio [num_layers, num_heads]
            hit_ratio_per_token [num_layers, num_heads, num_answer_tokens]
            top_k_indices [num_layers, num_heads, num_answer_tokens, top_k]
            top_k_scores  [num_layers, num_heads, num_answer_tokens, top_k]
        """
        num_layers, num_heads, num_answer_tokens, _ = attention_scores.shape

        tok_time_map = {
            t['token_idx']: {'word_start': t['word_start'], 'word_end': t['word_end']}
            for t in token_timestamps
        }

        hit_ratio_per_token = torch.zeros(num_layers, num_heads, num_answer_tokens)
        top_k_indices_all = torch.zeros(num_layers, num_heads, num_answer_tokens, self.top_k, dtype=torch.long)
        top_k_scores_all = torch.zeros(num_layers, num_heads, num_answer_tokens, self.top_k)

        for layer in range(num_layers):
            for head in range(num_heads):
                for tok_idx in range(num_answer_tokens):
                    if tok_idx not in tok_time_map:
                        continue
                    time_info = tok_time_map[tok_idx]
                    audio_start_idx = int(time_info['word_start'] * self.token_rate)
                    audio_end_idx = min(int(time_info['word_end'] * self.token_rate), num_audio)

                    attn_dist = attention_scores[layer, head, tok_idx]
                    if self.audio_only_topk:
                        top_k_scores, top_k_indices = torch.topk(attn_dist[:num_audio], self.top_k)
                    else:
                        top_k_scores, top_k_indices = torch.topk(attn_dist, self.top_k)

                    audio_mask = (top_k_indices >= audio_start_idx) & (top_k_indices < audio_end_idx)
                    hit_ratio = audio_mask.sum().float() / self.top_k

                    hit_ratio_per_token[layer, head, tok_idx] = hit_ratio
                    top_k_indices_all[layer, head, tok_idx] = top_k_indices
                    top_k_scores_all[layer, head, tok_idx] = top_k_scores

        valid_mask = torch.tensor(
            [tok_idx in tok_time_map for tok_idx in range(num_answer_tokens)],
            dtype=torch.bool,
        )
        hit_ratio = hit_ratio_per_token[:, :, valid_mask].mean(dim=2)

        return {
            "hit_ratio": hit_ratio,
            "hit_ratio_per_token": hit_ratio_per_token,
            "top_k_indices": top_k_indices_all,
            "top_k_scores": top_k_scores_all,
        }
