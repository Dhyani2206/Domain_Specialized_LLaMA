# Curated Knowledge, Better Models  
## Fine-Tuning LLaMA for Expert Data Science Reasoning

### Overview
This project explores whether carefully curated, domain-specific datasets can improve reasoning quality in Large Language Models.

We fine-tune:
- LLaMA-3 8B Instruct
- Mistral-7B Instruct
- Phi-3 Mini

using **QLoRA (4-bit quantization + LoRA adapters)** on ~8,000 high-quality Data Science Q&A pairs derived from textbooks and filtered academic sources.

The goal is to build a **domain-specialized AI Data Science Tutor**.

---

## Motivation
The original LLaMA paper emphasized that data quality outweighs scale.  
Public web datasets are noisy and inconsistent for academic reasoning tasks.

This project investigates:
- Can curated academic data improve reasoning?
- How do small vs larger models compare under identical fine-tuning?
- Does more prompting (5-shot) always help?

---

## Methodology

**Pipeline:**

Curated Dataset → Tokenization → QLoRA Fine-Tuning → Semantic Evaluation → RAG-based Inference

### Training Configuration
- Quantization: 4-bit (QLoRA)
- LoRA Rank (r): 16
- Alpha: 32
- Learning Rate: 1.5e-4
- Epochs: 2
- Sequence Length: 512
- Optimizer: AdamW
- Loss: Cross Entropy

---

## Evaluation

Metrics used:
- ROUGE-L
- BERTScore F1
- Semantic Accuracy (F1 > 0.8 threshold)
- Zero-shot, One-shot, Five-shot prompting

### Key Findings
- LLaMA-3 8B showed strongest zero-shot reasoning.
- Mistral-7B delivered stable, competitive performance.
- Phi-3 improved with 1-shot prompting.


## Application
A domain-specialized AI Data Science Tutor capable of:
- Explaining ML & statistical concepts
- Structured technical reasoning
- Reduced hallucination via RAG
- Supporting students & self-learners



## Future Work
- Expand curated dataset
- Layer-wise LR decay
- RLHF integration
- Active learning feedback loop
- Improved RAG retrieval
