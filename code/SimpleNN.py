import tensorflow as tf
import keras
import simpleGPTDict as eds

# Model parameters
embedding_dim = 128           # Size of the embedding vectors
sequence_length = 32          # Maximum sequence length for input and output
vocab_size = 2000            # Size of the vocabulary (maximum number of unique tokens)

# Training parameters
batch_size_set = 64           # Batch size (number of samples per update)
epoch_set = 200               # Number of epochs (iterations over the entire dataset)
learning_rate_set = 0.0025   # Initial learning rate for the optimizer

# Path for saving/loading the model
model_path = "C:\\Projects\\SimpleGPT\\code\\"

# Data preparation: Format the dataset into questions and answers with special tokens
questions = ["starttoken " + pair[0] + " endtoken" for pair in eds.dataSet]
answers = ["starttoken " + pair[1] + " endtoken" for pair in eds.dataSet]

# Tokenization
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,  # Limit the vocabulary size
    output_mode='int',      # Output integer tokens for each word
    output_sequence_length=sequence_length  # Pad sequences to a fixed length
)

# Adapt the tokenizer to the combined questions and answers dataset
tokenizer.adapt(questions + answers)
vocab_size = len(tokenizer.get_vocabulary())  # Update the vocabulary size based on the tokenizer

# Tokenize the questions and answers
questions_tokenized = tokenizer(questions)
answers_tokenized = tokenizer(answers)

# Padding sequences to ensure uniform length for input and output sequences
questions_padded = tf.keras.preprocessing.sequence.pad_sequences(questions_tokenized, padding='post')
answers_padded = tf.keras.preprocessing.sequence.pad_sequences(answers_tokenized, padding='post')

# Model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length),  # Embedding layer
    tf.keras.layers.GRU(1024, return_sequences=True),  # GRU layer for sequence processing
    tf.keras.layers.Dense(1024, activation='relu'),  # Fully connected layer with ReLU activation
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))  # Output layer for each time step
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_set),  # Adam optimizer with custom learning rate
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Sparse categorical crossentropy loss (for integer targets)
    metrics=['accuracy']  # Track accuracy during training
)

# Model training or loading
mode = input("Make choice: 1 - Train model, 0 - Load pre-trained model:\n")

if mode == '0':
    # Load the pre-trained model if mode is 0
    model = keras.models.load_model(f"{model_path}simple_nn_model.keras")
    model.summary()  # Show the model architecture
else:
    # Train the model if mode is 1
    model.fit(questions_padded, answers_padded, epochs=epoch_set, batch_size=batch_size_set)  # Train the model
    # Save the trained model to disk
    model.save(f"{model_path}simple_nn_model.keras")
    print(f"Model saved to {model_path}simple_nn_model.keras")

# Function to generate a response based on input text
def generate_response(input_text):
    input_tokens = tokenizer([input_text])  # Tokenize the input text
    predictions = model(input_tokens)      # Get model predictions for the input
    predicted_ids = tf.argmax(predictions[0], axis=-1).numpy()  # Get the most likely token IDs

    print(f"Predicted token IDs: {predicted_ids}")  # Log predicted token IDs for debugging
    response_tokens = []
    for idx in predicted_ids:
        word = tokenizer.get_vocabulary()[idx]  # Convert token IDs back to words
        if word == "endtoken":  # Stop when the 'endtoken' is reached
            break
        response_tokens.append(word)
    return " ".join(response_tokens).replace("starttoken", "").strip()  # Return the response as a string

# Interactive mode to continuously ask questions and generate responses
while True:
    question = "starttoken " + input("Question: ") + " endtoken"  # Format the question with special tokens
    response = generate_response(question)  # Generate a response based on the question
    print(f"Response: {response}")  # Print the generated response
