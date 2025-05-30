import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datasets import load_dataset

from DPO import OnlineDPOTrainer

raw = load_dataset("argilla/ultrafeedback-binarized-preferences", split="train[:100]")
prompts = raw["instruction"]  
print(prompts)

model_name = "arnir0/Tiny-LLM" 
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name)
ref_model  = AutoModelForCausalLM.from_pretrained(model_name) 

# ensure pad token
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))
    ref_model.resize_token_embeddings(len(tokenizer))

def collate_prompts(batch: list[str]) -> dict:
    return {"prompt": batch}

loader = DataLoader(
    prompts,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_prompts,
)

class RandomJudge:
    def judge(self, prompts, pairs):
        # returns 0 if first completion wins, 1 otherwise
        return torch.randint(0, 2, (len(pairs),)).tolist()

judge = RandomJudge()

# Instantiate the trainer
trainer = OnlineDPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    judge=judge,
    data_collator=lambda examples: tokenizer(examples, padding=True, return_tensors="pt"),
    generation_config=None,   # use defaults or pass a GenerationConfig
    max_length=256,
    args=type("A", (), {
        "loss_type": "sigmoid",
        "beta": 0.1,
        "n_gpu": 1,
        "gradient_accumulation_steps": 1,
        "torch_empty_cache_steps": None,
        "optim": None,
    })()
)

# Optimizer & scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
total_steps = len(loader) * 3  # e.g. 3 epochs
sched = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

# Move models to device
device = "cuda" if torch.cuda.is_available() else "mps"
model.to(device)
ref_model.to(device)

# Set models to appropriate modes
model.train()
ref_model.eval()  # Reference model should be in eval mode

# Online training loop
for epoch in range(3):
    for step, batch in enumerate(loader):
        optimizer.zero_grad()  # Zero gradients BEFORE forward pass
        
        # The training_step method handles forward pass, loss computation, and backward pass
        loss = trainer.training_step(model, batch)
        
        # NO NEED TO CALL loss.backward() here - it's already done in training_step!
        # The returned loss is already detached and just for logging
        
        optimizer.step()
        sched.step()

        if step % 50 == 0:
            print(f"Epoch {epoch+1} step {step}/{len(loader)} — loss: {loss.item():.4f}")

print("Training completed!")