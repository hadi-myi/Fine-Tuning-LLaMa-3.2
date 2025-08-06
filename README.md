Project Summary

This project explores fine-tuning pretrained open-source Large Language Models (LLMs) on two tasks where the base models struggle: language translation (English-Spanish) and LeetCode-style Python coding problems. Alongside this, I built and trained a character-level GPT transformer model from scratch to compare transfer learning against training from zero.

I fine-tuned Meta’s LLaMA 3.2 1B model using efficient techniques like LoRA and 4-bit quantization to make the process accessible on limited hardware. While the custom GPT trained from scratch showed some pattern learning, its output was mostly incoherent due to dataset size and compute constraints.

The fine-tuned translation model performed well, achieving a BLEU score around 0.4 (compared to ~0.15 for the base model), showing fine-tuning’s practical benefits even with limited resources. However, fine-tuning for code generation was less successful - the model improved formatting and syntax mimicry but mostly failed on logic and problem solving.

Overall, this project was a solid learning experience on transfer learning, fine-tuning techniques, and LLM internals. It highlighted how task complexity and data quality affect fine-tuning outcomes. Future work could involve better datasets and longer training to push performance further.
