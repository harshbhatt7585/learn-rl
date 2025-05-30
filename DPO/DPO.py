import torch
import torch.nn.functional as F
from typing import Union, Optional, Any
from torch import nn
from types import SimpleNamespace


def first_true_indices(bools: torch.Tensor, dtype: torch.dtype = torch.long) -> torch.Tensor:
    """
    Returns the index of the first True value in each row of a boolean tensor.
    If no True is found, returns the number of columns (i.e., len of row).
    """
    row_len = bools.size(-1)
    # Convert boolean mask to 0/1 then scale
    inv = (~bools).to(dtype)
    arange = torch.arange(row_len, dtype=dtype, device=bools.device)
    zero_or_index = inv * row_len + arange
    return torch.min(zero_or_index, dim=-1).values


def truncate_right(
    input_ids: torch.Tensor,
    stop_token_id: int,
    pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Truncates each sequence in `input_ids` at the first occurrence of `stop_token_id`.
    Tokens from the stop token onward are replaced by `pad_token_id`.
    Returns:
      - truncated_ids: Tensor with padded tokens after stop
      - attention_mask: 1 for tokens before the stop, 0 otherwise
    """
    # find first stop tokens per sequence (index of first True or seq_len)
    stops = input_ids == stop_token_id
    trunc_idxs = first_true_indices(stops)  # tensor of shape [...]

    # build index tensor for last dimension
    seq_len = input_ids.size(-1)
    shape = [1] * (input_ids.dim() - 1) + [seq_len]
    idxs = torch.arange(seq_len, device=input_ids.device).view(*shape)

    # mask positions from stop token (>=) onward
    right_mask = idxs >= trunc_idxs.unsqueeze(-1)

    # apply padding and create attention mask
    output_ids = input_ids.masked_fill(right_mask, pad_token_id)
    attention_mask = (~right_mask).long()

    return output_ids, attention_mask


class OnlineDPOTrainer:
    def __init__(
        self,
        model: Union[nn.Module, Any],  # simplify type hints
        ref_model: Optional[Union[nn.Module, Any]] = None,
        judge: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        generation_config: Optional[Any] = None,
        max_length: int = 1024,
        args: Optional[Any] = None,
    ):
        # core components
        self.model = model
        self.ref_model = ref_model if ref_model is not None else model
        self.judge = judge
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.max_length = max_length
        self.args = args or type("Args", (), {})()
        self.beta = 0.1
        # placeholder for training stats
        self.stats = {k: [] for k in [
            "val/contain_eos_token", "logps/chosen", "logps/rejected",
            "objective/kl", "objective/non_score_reward", "objective/entropy",
            "rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies", "beta"
        ]}

    def _prepare_inputs(self, batch: dict) -> dict:
        """
        Convert collated batch into model-ready inputs.
        Expects batch to have 'input_ids' and 'attention_mask'.
        Returns dict with 'prompt_input_ids' and 'prompt_attention_mask'.
        """
        return {
            'prompt_input_ids': batch['input_ids'],
            'prompt_attention_mask': batch['attention_mask'],
        }
    def generate(self, model: nn.Module, prompts: list[str]):
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        # 1) Directly tokenize raw prompts
        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # 2) Duplicate for pairwise generation
        prompt_ids  = batch["input_ids"].repeat(2, 1)
        prompt_mask = batch["attention_mask"].repeat(2, 1)

        # 3) Generate two completions per prompt
        out = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            generation_config=self.generation_config,
        )

        # 4) Strip off prompts and truncate at eos
        comps = out[:, prompt_ids.size(1):]
        comp_ids, comp_mask = truncate_right(comps, eos_id, pad_id)

        return prompt_ids, prompt_mask, comp_ids, comp_mask


    def _forward(self, model: nn.Module, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor,
                 completion_ids: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
        # Handle long sequences by truncating left
        total_len = prompt_ids.size(1) + completion_ids.size(1)
        excess = max(total_len - self.max_length, 0)
        if excess > 0:
            prompt_ids = prompt_ids[:, excess:]
            prompt_mask = prompt_mask[:, excess:]

        # concatenate and compute logprobs
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        outputs = model(input_ids, attention_mask=attention_mask)
        # align logits to completions
        logits = outputs.logits[:, prompt_ids.size(1)-1:-1]
        log_probs = F.log_softmax(logits, dim=-1)
        # gather logprobs for actual tokens
        gathered = torch.gather(log_probs, 2, completion_ids.unsqueeze(-1)).squeeze(-1)
        return gathered

    def training_step(self, model: nn.Module, inputs: dict[str, Any], num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        model.train()
        prompts = inputs["prompt"]
        batch_size = len(prompts)

        prompt_ids, prompt_mask, completion_ids, completion_mask = self.generate(model, prompts)
        contain_eos = (completion_ids == self.tokenizer.eos_token_id).any(dim=-1)

        logprobs = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        with torch.no_grad():
            ref_logprobs = self._forward(self.ref_model, prompt_ids, prompt_mask,
                                          completion_ids, completion_mask)

        # decode and judge
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        print(completions)
        ranks = self.judge.judge(
            prompts, list(zip(completions[:batch_size], completions[batch_size:]))
        ) if self.judge else [0]*batch_size
        device = logprobs.device
        mask = torch.tensor([r==0 for r in ranks], device=device)

        # select chosen and rejected
        idx = torch.arange(batch_size, device=device)
        chosen = idx + (~mask*batch_size)
        rejected = idx + (mask*batch_size)
        cr_idx = torch.cat([chosen, rejected], dim=0)

        lp = logprobs[cr_idx]
        ref_lp = ref_logprobs[cr_idx]
        padding = ~completion_mask.bool()[cr_idx]

        sums = lambda x: (x * ~padding).sum(1)
        pi_c, pi_r = sums(lp).split(batch_size)
        ref_c, ref_r = sums(ref_lp).split(batch_size)

        pi_lr = pi_c - pi_r
        ref_lr = ref_c - ref_r
        logits = pi_lr - ref_lr

        # compute loss
        loss_type = self.args.loss_type
        if loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif loss_type == "ipo":
            losses = (logits - 1/(2*self.beta))**2
        else:
            raise NotImplementedError(f"invalid loss type {loss_type}")
        loss = losses.mean()

        # backprop
        if self.args.n_gpu > 1:
            loss = loss.mean()
        loss.backward()
        return loss.detach() / getattr(self.args, 'gradient_accumulation_steps', 1)


if __name__ == "__main__":

    bools = torch.tensor([[False, True, False], [False, False, False]])
    idx = first_true_indices(bools)
    assert torch.equal(idx, torch.tensor([1, 3])), f"Got {idx.tolist()}"

    ids = torch.tensor([[1,2,3,99,5],[6,7,8,9,10]])
    out_ids, mask = truncate_right(ids, stop_token_id=99, pad_token_id=0)
    expected_ids = torch.tensor([[1,2,3,0,0],[6,7,8,9,10]])
    expected_mask = torch.tensor([[1,1,1,0,0],[1,1,1,1,1]])
    assert torch.equal(out_ids, expected_ids), f"IDs: {out_ids}"
    assert torch.equal(mask, expected_mask), f"Mask: {mask}"

    class DummyModel(nn.Module):
        def __init__(self, vocab_size=50, emb_dim=32):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, emb_dim)
            self.to_logits = nn.Linear(emb_dim, vocab_size)

        def generate(self, input_ids, attention_mask=None, generation_config=None):
            batch, seq_len = input_ids.size()
            eos_tokens = torch.full((batch, 1), 2, device=input_ids.device, dtype=input_ids.dtype)
            return torch.cat([input_ids, eos_tokens], dim=1)

        def forward(self, input_ids, attention_mask=None):
            x = self.embed(input_ids)                      # [batch, seq, emb_dim]
            logits = self.to_logits(x)                     # [batch, seq, vocab_size]
            return SimpleNamespace(logits=logits)
    
    class DummyTokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def apply_chat_template(self, x):
            return x

        def batch_decode(self, toks, skip_special_tokens=True):
            return ["".join(map(str, row.tolist())) for row in toks]

        def __call__(self, prompts):
            # For simplicity, give every prompt the same fake token IDs [1,2,3]
            batch_size = len(prompts)
            seq = torch.tensor([1, 2, 3], dtype=torch.long)
            input_ids = seq.unsqueeze(0).expand(batch_size, -1).clone()
            attention_mask = torch.ones_like(input_ids)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            
    def dummy_collator(batch):
        return {
            "prompt_input_ids": torch.stack([torch.tensor([1,2,3]) for _ in batch]),
            "prompt_attention_mask": torch.stack([torch.tensor([1,1,1]) for _ in batch]),
        }

    class DummyJudge:
        def judge(self, prompts, pairs): return [0 for _ in pairs]

    Args = type("Args", (), {
        "loss_type": "sigmoid",
        "n_gpu": 1,
        "gradient_accumulation_steps": 1,
        "torch_empty_cache_steps": None,
        "optim": None,
    })

    trainer = OnlineDPOTrainer(
        model=DummyModel(),
        ref_model=None,
        judge=DummyJudge(),
        data_collator=dummy_collator,
        tokenizer=DummyTokenizer(),
        generation_config=None,
        max_length=10,
        args=Args()
    )

    dummy_inputs = {"prompt": ["hello", "world"]}
    loss = trainer.training_step(trainer.model, dummy_inputs)
    print(f"(loss={loss.item():.4f})")

    print("\nAll checks passed!")
