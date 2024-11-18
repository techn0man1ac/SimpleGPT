import tensorflow as tf # Importing TensorFlow library for deep learning models
import numpy as np # Importing NumPy for numerical operations
import simpleGPTDict as eds # Import the dataset from simpleGPTDict

# Model parameters
embedding_dim = 128      # Dimension of word embeddings
num_heads = 4            # Number of attention heads in the multi-head attention layer
ff_dim = 1024            # Dimension of the feed-forward network
num_layers = 4           # Number of transformer layers

# Training parameters
epochs = 200             # Number of training epochs
batch_size = 64          # Number of samples per batch during training
lerning_rate = 0.000765  # Learning rate for the optimizer
sequence_length = 32     # Maximum sequence length (padding/truncation)

# Create a tokenizer to convert text into integer sequences
tokenizer = tf.keras.layers.TextVectorization(output_mode='int', output_sequence_length=sequence_length)

# Extracting questions and answers from the dataset (assuming 'dataSet' is a list of pairs)
questions = [pair[0] for pair in eds.dataSet]
answers = [pair[1] for pair in eds.dataSet]

# Adapt the tokenizer to the dataset (learns vocabulary from the questions and answers)
tokenizer.adapt(questions + answers)

# Function to tokenize and pad text sequences
def tokenize_and_pad(texts):
    return tokenizer(texts)

# Tokenizing and padding questions and answers
questions_tokenized = tokenize_and_pad(questions)
answers_tokenized = tokenize_and_pad(answers)

# Padding sequences to ensure they all have the same length
questions_padded = tf.keras.preprocessing.sequence.pad_sequences(questions_tokenized, padding='post')
answers_padded = tf.keras.preprocessing.sequence.pad_sequences(answers_tokenized, padding='post')

# Define the Transformer model class
class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_heads, ff_dim, num_layers):
        super(TransformerModel, self).__init__()
        
        # Word embedding layer
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        
        # Positional encoding layer to add positional information to the embeddings
        self.pos_encoding = self.get_positional_encoding(100000, embedding_dim)
        
        # Create multiple encoder layers (list of layers)
        self.encoder_layers = [self.create_encoder_layer(embedding_dim, num_heads, ff_dim) for _ in range(num_layers)]
        
        # Final output dense layer
        self.dense = tf.keras.layers.Dense(vocab_size)

    # Function to generate positional encodings
    def get_positional_encoding(self, max_len, embed_size):
        position = np.arange(max_len)[:, np.newaxis]  # Shape: (max_len, 1)
        div_term = np.exp(np.arange(0, embed_size, 2) * -(np.log(100000.0) / embed_size))  # Scaling factor
        pos_enc = np.zeros((max_len, embed_size))  # Initialize positional encoding matrix
        pos_enc[:, 0::2] = np.sin(position * div_term)  # Even indices use sin
        pos_enc[:, 1::2] = np.cos(position * div_term)  # Odd indices use cos
        return tf.constant(pos_enc, dtype=tf.float32)

    # Create a single encoder layer: Multi-head Attention + Feed-forward network
    def create_encoder_layer(self, embed_size, num_heads, ff_dim):
        attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_size)  # Multi-head self-attention
        ffn = tf.keras.Sequential([  # Feed-forward network with ReLU activation
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_size)
        ])
        return [attention, ffn]

    # Forward pass through the transformer model
    def call(self, inputs):
        # Add embedding and positional encoding
        x = self.embedding(inputs) + self.pos_encoding[:tf.shape(inputs)[1], :]
        
        # Pass through each encoder layer
        for attention, ffn in self.encoder_layers:
            # Self-attention step
            attn_output = attention(x, x)  # Attention mechanism (queries, keys, values are all the same)
            x = tf.keras.layers.LayerNormalization()(x + attn_output)  # Add & Norm
            
            # Feed-forward network
            ffn_output = ffn(x)
            x = tf.keras.layers.LayerNormalization()(x + ffn_output)  # Add & Norm
        
        return self.dense(x)  # Final dense layer to output predicted token logits

# Vocabulary size is based on the tokenizer's vocabulary
vocab_size = len(tokenizer.get_vocabulary())

# Initialize the transformer model
model = TransformerModel(vocab_size, embedding_dim, num_heads, ff_dim, num_layers)

# Define loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')  # Sparse Categorical Crossentropy loss
optimizer = tf.keras.optimizers.Adam(lerning_rate)  # Adam optimizer with a custom learning rate

# Accuracy metric to track model performance during training
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Training step function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:  # Track gradients for backpropagation
        predictions = model(inputs)  # Forward pass to get model predictions
        loss = loss_fn(targets, predictions)  # Compute loss (difference between predictions and true targets)
        loss = tf.reduce_mean(loss)  # Average the loss over the batch
    
    gradients = tape.gradient(loss, model.trainable_variables)  # Compute gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Apply gradients to update model weights
    
    # Update accuracy metric
    accuracy_metric.update_state(targets, predictions)
    
    return loss

# Training loop over epochs
for epoch in range(epochs):
    # Reset accuracy metric at the beginning of each epoch
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for i in range(0, len(questions_padded), batch_size):
        batch_input = questions_padded[i:i+batch_size]  # Get a batch of input questions
        batch_target = answers_padded[i:i+batch_size]  # Get the corresponding batch of answers
        
        loss = train_step(batch_input, batch_target)  # Perform one training step
    
    accuracy = accuracy_metric.result().numpy()  # Compute the accuracy for the epoch
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.numpy():.5f}, Accuracy: {accuracy:.3f}')

# Function to decode token indices into text (ignores padding tokens)
def decode_tokens(tokens, tokenizer):
    # Get reverse vocabulary (mapping from token indices to words)
    reverse_vocab = tokenizer.get_vocabulary()
    
    # Convert token indices back to words, skipping padding tokens (0)
    return ' '.join([reverse_vocab[token] for token in tokens if token != 0])

# Function to generate a response given a question
def generate_response(question, tokenizer, model):
    # Tokenize and pad the input question
    tokens = tokenize_and_pad([question])
    tokens = tf.convert_to_tensor(tokens)  # Convert to tensor
    
    # Generate model predictions (sequence of token probabilities)
    generated = model(tokens)
    
    # Extract the most probable token indices (argmax)
    generated_ids = tf.argmax(generated, axis=-1).numpy()[0]
    
    # Decode token indices into a human-readable response
    response = decode_tokens(generated_ids, tokenizer)
    return response

# Interactive loop for testing the model: keep asking for questions and get responses
while(True):
    question = input("Question: ")  # Ask the user for input
    response = generate_response(question, tokenizer, model)  # Generate the response
    print(f"Response: {response}")  # Print the model's response
