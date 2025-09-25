import tensorflow as tf
import pickle
from Transformer2 import Transformer  # Make sure this imports your Transformer class


def load_trained_model(weights_path, tokenizer_path):
    """
    Load trained Transformer model and tokenizer
    """
    print("Loading tokenizer...")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    print("Creating model...")
    model = Transformer(
        num_layers=4,
        embedding_dim=128,
        num_heads=8,  
        fully_connected_dim=512,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        max_positional_encoding_input=200,
        max_positional_encoding_target=200,
        dropout_rate=0.3 
    )
    
    print("Building model...")
    dummy_input = tf.ones((1, 10), dtype=tf.int64)
    dummy_output = tf.ones((1, 10), dtype=tf.int64)
    _ = model(dummy_input, dummy_output, training=False)
    
    print("Loading weights...")
    model.load_weights(weights_path)
    
    print("Model loaded successfully!")
    return model, tokenizer


def sample_from_logits(logits, top_k=0, top_p=0.0):
    """
    Apply top-k and/or nucleus (top-p) sampling to logits.
    """
    logits = tf.squeeze(logits)

    # Top-k filtering
    if top_k > 0:
        values, _ = tf.math.top_k(logits, k=top_k)
        min_values = values[-1]
        logits = tf.where(logits < min_values, float("-inf"), logits)

    # Top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits = tf.sort(logits, direction="DESCENDING")
        cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits), axis=-1)
        cutoff = tf.reduce_min(tf.boolean_mask(sorted_logits, cumulative_probs <= top_p))
        logits = tf.where(logits < cutoff, float("-inf"), logits)

    # Sample from filtered distribution
    probs = tf.nn.softmax(logits)
    predicted_id = tf.random.categorical([probs], num_samples=1)
    return tf.cast(predicted_id, tf.int64)


def generate_text_word_level(model, tokenizer, start_string, max_length=50,
                             temperature=0.8, top_k=0, top_p=0.0):
    """
    Generate text at word-level using trained model with temperature,
    top-k, and nucleus (top-p) sampling.
    """
    print(f"Generating text starting with: '{start_string}'")

    # Convert start string to sequence of word IDs
    start_words = start_string.lower().split()
    input_seq = tokenizer.texts_to_sequences([start_words])[0]

    if not input_seq:
        print("Warning: No valid tokens from start string. Using <OOV> instead.")
        input_seq = [tokenizer.word_index.get("<OOV>", 1)]

    input_tensor = tf.expand_dims(tf.cast(input_seq, tf.int64), 0)
    output = tf.identity(input_tensor)

    for i in range(max_length):
        predictions, _ = model(
            input_sentence=output,
            output_sentence=output,
            training=False
        )

        # Apply temperature
        predictions = predictions[:, -1, :] / temperature

        # Use top-k / top-p sampling
        predicted_id = sample_from_logits(predictions, top_k=top_k, top_p=top_p)

        if predicted_id[0, 0] == 0:  # stop if padding
            break

        output = tf.concat([output, predicted_id], axis=-1)

    # Convert IDs back to words
    generated_sequence = output[0].numpy()
    generated_words = [tokenizer.index_word.get(i, "<UNK>") for i in generated_sequence if i != 0]

    return " ".join(generated_words)


if __name__ == "__main__":

    weights_path = "transformer_sonnet_model_final.weights.h5"
    tokenizer_path = "transformer_sonnet_model_tokenizer2.pkl"
    Input = input("Enter the starting prompt : ")
    
    try:
        model, tokenizer = load_trained_model(weights_path, tokenizer_path)

        # Example: nucleus sampling with top_p=0.9
        generated = generate_text_word_level(
            model, tokenizer, Input,
            max_length=100, temperature=0.2, top_k=40, top_p=0.9
        )

        print("\nGenerated text:\n")
        print(generated)

    except Exception as e:
        print(f"Error loading model: {e}")
