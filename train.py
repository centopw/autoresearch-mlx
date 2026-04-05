"""
Autoresearch pretraining script. Single-device (Apple Silicon MLX), single-file.
Cherry-picked and simplified from nanochat. Ported from PyTorch/CUDA to MLX.
Usage: uv run train.py
"""

import gc
import math
import time
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return mx.fast.rms_norm(x, weight=None, eps=1e-5)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    # x: (B, T, n_head, head_dim)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=3)


def _sliding_window_causal_mask(T, window):
    """Additive attention mask: 0.0 where allowed, -inf where masked."""
    rows = mx.arange(T)[:, None]   # (T, 1)
    cols = mx.arange(T)[None, :]   # (1, T)
    causal = cols <= rows
    window_ok = (rows - cols) < window
    allowed = causal & window_ok
    return mx.where(allowed, mx.zeros((T, T)), mx.full((T, T), float('-inf')))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
                        if has_ve(layer_idx, config.n_layer) else None)
        self._layer_idx = layer_idx

    def __call__(self, x, ve, cos_sin, window_size):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer)
        if ve is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate[..., None] * ve

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = norm(q)
        k = norm(k)

        # MLX SDPA expects (B, n_head, T, head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        scale = self.head_dim ** -0.5
        window = window_size[0]
        if window >= T:
            # Full causal attention
            mask = mx.tril(mx.ones((T, T)))
            mask = mx.where(mask, mx.zeros((T, T)), mx.full((T, T), float('-inf')))
        else:
            mask = _sliding_window_causal_mask(T, window)

        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        y = mx.transpose(y, (0, 2, 1, 3)).reshape(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.square(nn.relu(x))
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones((config.n_layer,))
        self.x0_lambdas = mx.zeros((config.n_layer,))
        # Value embeddings (plain dict — MLX auto-discovers params in dicts)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        }
        # Rotary embeddings (stored as plain attributes, not parameters)
        self.rotary_seq_len = config.sequence_len * 10
        self.cos, self.sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim)

    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        # Embedding and unembedding
        self.wte.weight = mx.random.normal(shape=self.wte.weight.shape)
        self.lm_head.weight = mx.random.normal(shape=self.lm_head.weight.shape) * 0.001
        # Transformer blocks
        for block in self.h:
            block.attn.c_q.weight = mx.random.uniform(-s, s, shape=block.attn.c_q.weight.shape)
            block.attn.c_k.weight = mx.random.uniform(-s, s, shape=block.attn.c_k.weight.shape)
            block.attn.c_v.weight = mx.random.uniform(-s, s, shape=block.attn.c_v.weight.shape)
            block.attn.c_proj.weight = mx.zeros(block.attn.c_proj.weight.shape)
            block.mlp.c_fc.weight = mx.random.uniform(-s, s, shape=block.mlp.c_fc.weight.shape)
            block.mlp.c_proj.weight = mx.zeros(block.mlp.c_proj.weight.shape)
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight = mx.zeros(block.attn.ve_gate.weight.shape)
        # Per-layer scalars
        self.resid_lambdas = mx.ones((self.config.n_layer,))
        self.x0_lambdas = mx.full((self.config.n_layer,), 0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-s, s, shape=ve.weight.shape)
        # Cast embeddings to bfloat16
        self.wte.weight = self.wte.weight.astype(mx.bfloat16)
        for ve in self.value_embeds.values():
            ve.weight = ve.weight.astype(mx.bfloat16)
        # Refresh rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        mx.eval(self.parameters())

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        cos = mx.cos(freqs).astype(mx.bfloat16)[None, :, None, :]  # (1,T,1,head_dim//2)
        sin = mx.sin(freqs).astype(mx.bfloat16)[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.size for p in mx.utils.tree_flatten(self.parameters()))
        ve_params = sum(p.size for p in mx.utils.tree_flatten(
            [ve.parameters() for ve in self.value_embeds.values()]))
        wte_params = self.wte.weight.size
        scalar_params = self.resid_lambdas.size + self.x0_lambdas.size
        nparams_exclude = wte_params + ve_params + scalar_params
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = self.wte.weight.size
        value_embeds = sum(p.size for p in mx.utils.tree_flatten(
            [list(ve.parameters().values()) for ve in self.value_embeds.values()]))
        lm_head = self.lm_head.weight.size
        transformer_matrices = sum(
            p.size for p in mx.utils.tree_flatten([b.parameters() for b in self.h]))
        scalars = self.resid_lambdas.size + self.x0_lambdas.size
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Collect named parameters
        named_params = dict(mx.utils.tree_flatten(self.trainable_parameters()))

        # Identify param groups by path
        matrix_params = []
        lm_head_params = []
        embedding_params = []
        value_embeds_params = []
        resid_params = []
        x0_params = []

        for path, p in named_params.items():
            if path == 'resid_lambdas':
                resid_params.append((path, p))
            elif path == 'x0_lambdas':
                x0_params.append((path, p))
            elif path.startswith('lm_head'):
                lm_head_params.append((path, p))
            elif path.startswith('wte'):
                embedding_params.append((path, p))
            elif path.startswith('value_embeds'):
                value_embeds_params.append((path, p))
            else:
                matrix_params.append((path, p))

        param_groups = [
            dict(kind='adamw', params=lm_head_params,
                 lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params,
                 lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params,
                 lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params,
                 lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params,
                 lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Group matrix params by shape for Muon
        shapes = sorted({p.shape for _, p in matrix_params})
        for shape in shapes:
            group_params = [(path, p) for path, p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def __call__(self, idx, targets=None, reduction='mean'):
        B, T = idx.shape
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15.0
        logits = self.lm_head(x).astype(mx.float32)
        logits = softcap * mx.tanh(logits / softcap)

        if targets is not None:
            V = logits.shape[-1]
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, V), targets.reshape(-1), reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class MuonAdamW:
    """Combined optimizer: Muon for 2D matrix params, AdamW for others.
    param_groups: list of dicts with keys 'kind', 'params' (list of (path, array)), 'lr', etc.
    """

    def __init__(self, param_groups):
        self.param_groups = param_groups
        self.state = {}  # keyed by param path string

    def _adamw_step(self, path, p, grad, group):
        state = self.state.setdefault(path, {})
        state['step'] = state.get('step', 0) + 1
        step = state['step']
        if 'exp_avg' not in state:
            state['exp_avg'] = mx.zeros_like(p)
            state['exp_avg_sq'] = mx.zeros_like(p)
        lr = group['lr']
        b1, b2 = group['betas']
        eps = group['eps']
        wd = group['weight_decay']
        exp_avg = (1 - b1) * grad + b1 * state['exp_avg']
        exp_avg_sq = (1 - b2) * mx.square(grad) + b2 * state['exp_avg_sq']
        state['exp_avg'] = exp_avg
        state['exp_avg_sq'] = exp_avg_sq
        bias1 = 1 - b1 ** step
        bias2 = 1 - b2 ** step
        denom = mx.sqrt(exp_avg_sq / bias2) + eps
        step_size = lr / bias1
        return p * (1 - lr * wd) - step_size * exp_avg / denom

    def _muon_step(self, group, grads_dict):
        params_list = group['params']  # list of (path, array)
        if not params_list:
            return {}

        path0, p0 = params_list[0]
        shape = p0.shape
        num_params = len(params_list)

        # Stack grads and params
        stacked_grads = mx.stack([grads_dict[path] for path, _ in params_list])  # (N, *shape)
        stacked_params = mx.stack([p for _, p in params_list])

        state = self.state.setdefault(f'muon_{shape}', {})

        # Nesterov momentum
        m = group['momentum']
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = mx.zeros_like(stacked_grads)
        buf = (1 - m) * stacked_grads + m * state['momentum_buffer']
        state['momentum_buffer'] = buf
        g = (1 - m) * stacked_grads + m * buf  # Nesterov

        # Newton-Schulz orthogonalization (polar express)
        X = g.astype(mx.bfloat16)
        flat = X.reshape(num_params, shape[-2] * shape[-1]) if len(shape) == 2 else X.reshape(num_params, -1)
        norms = mx.linalg.norm(flat, axis=-1)[:, None, None] * 1.02 + 1e-6
        X = X / norms
        ns = group['ns_steps']
        if shape[-2] > shape[-1]:
            for a, b, c in polar_express_coeffs[:ns]:
                At = mx.transpose(X, (0, 2, 1))
                A = At @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
        else:
            for a, b, c in polar_express_coeffs[:ns]:
                A = X @ mx.transpose(X, (0, 2, 1))
                B = b * A + c * (A @ A)
                X = a * X + B @ X
        g = X.astype(stacked_grads.dtype)

        # NorMuon variance reduction
        beta2 = group['beta2']
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        v_mean = mx.square(g.astype(mx.float32)).mean(axis=red_dim, keepdims=True)
        red_dim_size = g.shape[red_dim]
        v_norm_sq = v_mean.sum(axis=(-2, -1), keepdims=True) * red_dim_size
        v_norm = mx.sqrt(v_norm_sq)
        if 'second_momentum_buffer' not in state:
            smb_shape = list(stacked_grads.shape)
            smb_shape[red_dim] = 1
            state['second_momentum_buffer'] = mx.zeros(smb_shape, dtype=stacked_grads.dtype)
        smb = (1 - beta2) * v_mean.astype(state['second_momentum_buffer'].dtype) + beta2 * state['second_momentum_buffer']
        state['second_momentum_buffer'] = smb
        step_size = mx.rsqrt(mx.maximum(smb.astype(mx.float32), 1e-10))
        scaled_sq_sum = (v_mean * red_dim_size) * mx.square(step_size)
        v_norm_new = mx.sqrt(scaled_sq_sum.sum(axis=(-2, -1), keepdims=True))
        final_scale = step_size * (v_norm / mx.maximum(v_norm_new, 1e-10))
        g = g * final_scale.astype(g.dtype)

        # Cautious weight decay + update
        lr = mx.array(group['lr'] * max(1.0, shape[-2] / shape[-1])**0.5, dtype=g.dtype)
        wd = mx.array(group['weight_decay'], dtype=g.dtype)
        mask = (g * stacked_params.astype(g.dtype)) >= 0
        new_params = stacked_params.astype(g.dtype) - lr * g - lr * wd * stacked_params.astype(g.dtype) * mask

        return {path: new_params[i] for i, (path, _) in enumerate(params_list)}

    def update(self, model, grads_flat):
        """Apply gradients. grads_flat: dict of {path: grad_array}."""
        new_params = {}

        for group in self.param_groups:
            if group['kind'] == 'adamw':
                for path, p in group['params']:
                    if path not in grads_flat:
                        continue
                    grad = grads_flat[path]
                    new_params[path] = self._adamw_step(path, p, grad, group)
            elif group['kind'] == 'muon':
                updated = self._muon_step(group, grads_flat)
                new_params.update(updated)

        # Update model parameters
        if new_params:
            current = dict(mx.utils.tree_flatten(model.trainable_parameters()))
            current.update(new_params)
            model.update(mx.utils.tree_unflatten(list(current.items())))

    def update_lrs(self, multiplier):
        for group in self.param_groups:
            group['lr'] = group['initial_lr'] * multiplier

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

TOTAL_BATCH_SIZE = 2**19
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

DEPTH = 8
DEVICE_BATCH_SIZE = 128

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")


def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

model = GPT(config)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)
mx.eval(x, y)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)


# Build loss+grad function once
def forward_loss(model, x, y):
    return model(x, y)

loss_and_grad_fn = nn.value_and_grad(model, forward_loss)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0.0
total_training_time = 0.0
step = 0

while True:
    t0 = time.time()

    accumulated_grads = None
    total_loss = 0.0
    for micro_step in range(grad_accum_steps):
        loss, grads = loss_and_grad_fn(model, x, y)
        mx.eval(loss)
        total_loss += loss.item()
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = mx.utils.tree_map(
                lambda a, b: a + b, accumulated_grads, grads)
        x, y, epoch = next(train_loader)
        mx.eval(x, y)

    train_loss_f = total_loss / grad_accum_steps

    # Fast fail
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    # Scale grads
    accumulated_grads = mx.utils.tree_map(
        lambda g: g / grad_accum_steps, accumulated_grads)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

    # Flatten grads for optimizer
    grads_flat = dict(mx.utils.tree_flatten(accumulated_grads))
    optimizer.update(model, grads_flat)
    mx.eval(model.parameters())

    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

t_end = time.time()
startup_time = t_start_training - t_start
total_training_time_actual = total_training_time

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time_actual:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     0.0")
print(f"mfu_percent:      0.00")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
