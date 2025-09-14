#install dependencies required
#```pip install transformers peft datasets accelerate bitsandbytes```
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# 1. Base model
model_name = "gpt2"  # (tiny for demo; jobs use Llama/Mistral etc.)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

# 2. Apply LoRA config
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. Toy dataset
data = {
    "text": [
        "Hello! -> Ahoy, matey!",
        "How are you? -> Arrr, I be good!",
    ]
}
dataset = Dataset.from_dict(data)

# 4. Tokenize
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
dataset = dataset.map(tokenize_fn)

# 5. Training args
args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=1
)

# 6. Trainer
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()

# 7. Save adapter weights
model.save_pretrained("./lora-pirate")


#run inference model with finetuned model
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "gpt2"
base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "./lora-pirate")
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(pipe("Hello!", max_new_tokens=20))
