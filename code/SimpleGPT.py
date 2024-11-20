import tensorflow as tf
import numpy as np
import keras
import simpleGPTDict as eds

# Model parameters
embedding_dim = 128           # Dimension of the word embeddings
num_heads = 4                 # Number of attention heads in Multi-Head Attention
ff_dim = 4096                 # Dimension of the feed-forward network (FFN)
num_layers = 4                # Number of transformer encoder layers
sequence_length = 32          # Maximum sequence length for input data
vocab_size = 10000            # Size of the vocabulary (update after tokenizer adaptation)

# Training parameters
batch_size_set = 64           # Batch size for training
epoch_set = 200               # Number of epochs to train
learning_rate_set = 0.000765  # Initial learning rate
decay_rate_set = 0.85         # Learning rate decay rate

# Path to save or load the model
model_path = "C:\\Projects\\vscode-basics\\GoIT-Python-Data-Science\\MyProjects\\SimpleGPT\\code\\"

# Dataset preparation
# Prepares the dataset by adding start and end tokens for questions and answers
questions = ["<start> " + pair[0] + " <end>" for pair in eds.dataSet]
answers = ["<start> " + pair[1] + " <end>" for pair in eds.dataSet]

# Tokenizer
# Converts text into sequences of integers with fixed sequence length
tokenizer = tf.keras.layers.TextVectorization(
    output_mode='int', output_sequence_length=sequence_length
)
tokenizer.adapt(questions + answers)  # Fit the tokenizer to the dataset

# Tokenized versions of questions and answers
questions_tokenized = tokenizer(questions)
answers_tokenized = tokenizer(answers)

# Positional Encoding
# Adds position-based information to embeddings to maintain order in sequences
def get_positional_encoding(max_len, embed_size):
    position = np.arange(max_len)[:, np.newaxis]  # Positions [0, 1, ..., max_len-1]
    div_term = np.exp(np.arange(0, embed_size, 2) * -(np.log(100000.0) / embed_size))
    pos_enc = np.zeros((max_len, embed_size))
    pos_enc[:, 0::2] = np.sin(position * div_term)  # Apply sine to even indices
    pos_enc[:, 1::2] = np.cos(position * div_term)  # Apply cosine to odd indices
    return tf.constant(pos_enc, dtype=tf.float32)

# Generate positional encoding matrix
positional_encoding = get_positional_encoding(sequence_length, embedding_dim)

# Functional API Model
inputs = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)  # Input layer

# Embedding layer
x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
x = x + positional_encoding  # Add positional encoding to embeddings

# Add transformer encoder layers
for _ in range(num_layers):
    # Multi-Head Attention
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embedding_dim
    )(x, x)  # Self-attention
    x = tf.keras.layers.LayerNormalization()(x + attention_output)  # Add & Normalize

    # Feed-Forward Network (FFN)
    ffn_output = tf.keras.layers.Dense(ff_dim, activation='relu')(x)  # First dense layer
    ffn_output = tf.keras.layers.Dense(embedding_dim)(ffn_output)    # Project back to embedding size
    x = tf.keras.layers.LayerNormalization()(x + ffn_output)         # Add & Normalize

# Final Dense layer to predict vocabulary tokens
outputs = tf.keras.layers.Dense(vocab_size)(x)

# Create the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
# Using exponential decay for the learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate_set, decay_steps=1000, decay_rate=decay_rate_set
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)  # Adam optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # Loss function

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Prepare padded sequences for training
questions_padded = tf.keras.preprocessing.sequence.pad_sequences(questions_tokenized, padding='post')
answers_padded = tf.keras.preprocessing.sequence.pad_sequences(answers_tokenized, padding='post')

# Train or load pre-trained model based on user input
mode = input("Make choice: 1 - Train model, 0 - Load pre-trained model:\n")

if mode == '0':
    # Load the pre-trained model
    model = keras.models.load_model(f"{model_path}simple_gpt_model.keras")
    model.summary()  # Display model summary
else:
    # Train the model
    model.fit(questions_padded, answers_padded, epochs=epoch_set, batch_size=batch_size_set)
    # Save the trained model
    model.save(f"{model_path}simple_gpt_model.keras")
    print(f"Model saved to {model_path}simple_gpt_model.keras")

# Function to decode token indices into text (ignores padding tokens)
def decode_tokens(tokens, tokenizer):
    reverse_vocab = tokenizer.get_vocabulary()  # Reverse mapping from indices to tokens
    return ' '.join([reverse_vocab[token] for token in tokens if token != 0])

# Function to generate a response given a question
def generate_response(input_text):
    input_tokens = tokenizer([input_text])  # Tokenize the input question
    predictions = model(input_tokens)      # Predict next tokens
    predicted_ids = tf.argmax(predictions[0], axis=-1).numpy()  # Get predicted token indices

    response_tokens = []
    for idx in predicted_ids:
        word = tokenizer.get_vocabulary()[idx]
        if word == "end":  # Stop generating when the end token is reached
            break
        response_tokens.append(word)
    return " ".join(response_tokens).replace("start", "").strip()  # Clean and format the response

# Interactive loop for testing the model
while True:
    question = "<start> " + input("Question: ") + " <end>"  # Format input question
    response = generate_response(question)                 # Generate model response
    print(f"Response: {response}")
