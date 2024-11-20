# Simple GPT Model

![Chat with model](https://raw.githubusercontent.com/techn0man1ac/SimpleGPT/refs/heads/main/img/Screenshot.png)

A simple implementation of a GPT-like model using TensorFlow and Keras. It uses a Transformer architecture with multi-head attention to generate responses from input questions.

## Features

- Transformer-based architecture with multi-head attention
- Positional encoding added to input embeddings
- Customizable hyperparameters (e.g., embedding size, number of layers)
- Tokenizer-based input processing
- Option to train from scratch or load a pre-trained model
- Interactive question-answer generation

## Requirements

Install dependencies:

```bash
pip install tensorflow keras numpy
```

# Dataset

The model uses a custom dataset (simpleGPTDict) with question-answer pairs. Each pair is tokenized with <start> and <end> tokens.

# Model Overview

Input: Tokenized question sequences
Architecture: Embedding → Positional Encoding → Transformer Encoder → Output Layer
Training: Uses SparseCategoricalCrossentropy loss and Adam optimizer

# Running the Model

## Train Model

To train the model, run:

```bash
python SimpleGPT.py
```

Select option `1` to train from scratch. The trained model will be saved as `simple_gpt_model.keras`.

## Load Pre-trained Model

To load an existing model, select option `0` when prompted.

# Usage

Interact with the model to generate responses:

```bash
Question: What is your name?
Response: i am eliks 
```

# License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/techn0man1ac/SimpleGPT/blob/main/LICENSE) file for details.
