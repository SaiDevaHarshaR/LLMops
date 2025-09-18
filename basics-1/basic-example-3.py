from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
Artificial Intelligence is rapidly changing the world. 
It is being used in healthcare, finance, education, and entertainment. 
While AI brings efficiency and new possibilities, it also raises ethical concerns 
such as job loss, bias, and privacy issues. 
Balancing innovation and responsibility is crucial for the future.
"""

summary = summarizer(article, max_length=50, min_length=20, do_sample=False)
print(summary[0]['summary_text'])
