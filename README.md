# AI Sonnet Generator

A Transformer-based text generation model that creates Shakespearean-style sonnets at the **word level**. Built with TensorFlow, this project trains on a collection of 154 sonnets and leverages **multi-head attention** and **positional encoding** for sequence modeling.

## Features
- Word-level Transformer for natural language generation
- Customizable generation with **temperature**, **top-k**, and **top-p (nucleus) sampling**
- Save and load trained model weights and tokenizer
- Generates creative sonnet-like text based on a user-provided prompt
- Since the model is trained on a small dataset the output is not accurate

## Requirements
- Python 3.10+
- TensorFlow 2.x
- Keras Preprocessing
- NumPy

Install dependencies with:
```bash
pip install -r requirements.txt
