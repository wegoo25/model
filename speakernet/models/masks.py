"""Padding-mask helpers used by the Transformer encoder."""
import torch  # noqa: F401  (kept for type context of tensor ops)


def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """Padding positions are set to 0; use either ``input_lengths`` or ``pad_idx``."""
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        n = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
        for i in range(n):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    return non_pad_mask.unsqueeze(-1)


def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """Mask positions are set to 1."""
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)  # N x Ti x 1
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    return pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
