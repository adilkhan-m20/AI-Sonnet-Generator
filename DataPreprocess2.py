import re
import numpy as np
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
import pickle
from Transformer2 import Transformer, create_padding_mask, create_look_ahead_mask

# Load sonnets
with open("Sonnet.txt","r",encoding="utf-8") as f:
    raw_text = f.read()
normalized_text = re.sub(r"\n\s*\n","\n\n",raw_text).strip()
sonnets = [s.strip() for s in normalized_text.split("\n\n") if s.strip()]
print(f"Total Sonnets: {len(sonnets)}")

# Word-level Tokenizer - FIXED
tokenizer = Tokenizer(lower=True, oov_token="<OOV>")
tokenizer.fit_on_texts(sonnets)
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding
print(f"Word vocabulary size: {vocab_size}")
print(f"Sample words: {list(tokenizer.word_index.items())[:10]}")

# Convert Sonnets to sequences
sequences = tokenizer.texts_to_sequences(sonnets)
all_tokens = [token for seq in sequences for token in seq]
print(f"Total Tokens: {len(all_tokens)}")

# Sliding windows at word-level
max_seq_len = 100  # shorter context works well at word-level
input_seq, target_seq = [], []
for i in range(0, len(all_tokens) - max_seq_len):
    input_sequence = all_tokens[i:i+max_seq_len]
    target_sequence = all_tokens[i+1:i+max_seq_len+1]
    input_seq.append(input_sequence)
    target_seq.append(target_sequence)

input_seq = np.array(input_seq)
target_seq = np.array(target_seq)
print(f"Input Shape: {input_seq.shape}, Target Shape: {target_seq.shape}")

# Tensorflow Datasets
BUFFER_SIZE = len(input_seq)
BATCH_SIZE = 32

train_dataset = tf.data.Dataset.from_tensor_slices((input_seq, target_seq)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Loss & Optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
def loss_function(real, pred):
    mask = tf.math.not_equal(real, 0)
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

optimizer = tf.keras.optimizers.Adam(1e-4)

# Transformer Model 
transformer_model = Transformer(
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

# Build the model with a sample input to initialize all layers
print("Building model...")
dummy_input = tf.ones((1, max_seq_len), dtype=tf.int64)
dummy_output = tf.ones((1, max_seq_len), dtype=tf.int64)
_ = transformer_model(dummy_input, dummy_output, training=False)
print("Model built successfully!")
print(f"Model has {transformer_model.count_params()} parameters")

# Training Step
@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
    dec_padding_mask = create_padding_mask(inp)
    combined_mask = tf.maximum(create_padding_mask(tar_inp), look_ahead_mask)

    with tf.GradientTape() as tape:
        predictions, _ = transformer_model(
            input_sentence=inp,
            output_sentence=tar_inp,
            training=True,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask
        )
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer_model.trainable_variables))
    return loss

# Training Loop
EPOCHS = 30
save_path = "transformer_sonnet_model"

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = 0
    num_train = 0

    # Training
    for (batch, (inp, tar)) in enumerate(train_dataset):
        batch_loss = train_step(inp, tar)
        train_loss += batch_loss
        num_train += 1
        if batch % 50 == 0:
            print(f"Batch {batch}, Train Loss {batch_loss.numpy():.4f}")

    avg_train_loss = train_loss / num_train
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")


# Save once at the end of training
print("\nSaving final model...")
transformer_model.save_weights(f"{save_path}_final.weights.h5")
with open(f"{save_path}_tokenizer2.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Final model and tokenizer saved!")