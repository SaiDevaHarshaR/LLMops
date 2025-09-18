from transformers import pipeline
gen = pipeline("text-generation", "distill-gpt2")
res = gen("AI is the future of", max_length=20, num_return_sequences=1)
print(res[0]['generated_text']) 