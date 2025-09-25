import numpy as np
import tensorflow as tf
#from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Input, Dropout, LayerNormalization

Embedding = tf.keras.layers.Embedding
MultiHeadAttention = tf.keras.layers.MultiHeadAttention
Dense = tf.keras.layers.Dense
Input = tf.keras.Input
Dropout = tf.keras.layers.Dropout
LayerNormalization = tf.keras.layers.LayerNormalization

def get_angles(pos,k,d):
    i = k//2
    angles = pos / np.power(10000,2*i/d)
    return angles

def positional_encoding(positions,d):
    angle_rads = get_angles(np.arange(positions)[:,np.newaxis], np.arange(d)[np.newaxis,:], d)
    angle_rads[:,0::2] = np.sin(angle_rads[:,0::2])
    angle_rads[:,1::2] = np.cos(angle_rads[:,1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype = tf.float32)

# Padding Mask
def create_padding_mask(decoder_token_ids):
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids,0), tf.float32)
    return seq[:,tf.newaxis,:]

# Look Ahead Mask
def create_look_ahead_mask(sequence_length):
    mask = tf.linalg.band_part(tf.ones((1,sequence_length,sequence_length)),-1,0)
    return mask

def scaled_dot_product_attention(q,k,v,mask):
    matmul_qk = tf.matmul(q,k, transpose_b = True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += ((1-mask)*-1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)
    output = tf.matmul(attention_weights,v)

    return output, attention_weights

def FullyConnected(embedding_dim,fully_connected_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim,activation='relu'),
        tf.keras.layers.Dense(embedding_dim)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,embedding_dim,num_heads,fully_connected_dim,dropout_rate=0.1,layernorm_eps=1e-6):
        super(EncoderLayer,self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads,key_dim=embedding_dim,dropout=dropout_rate)
        self.ffn = FullyConnected(embedding_dim=embedding_dim,fully_connected_dim=fully_connected_dim)
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout_ffn = Dropout(dropout_rate)
    def call(self,x,training=None,mask=None):
        self_mha_output = self.mha(x,x,x,attention_mask=mask,training=training)
        skip_x_attention = self.layernorm1(x+self_mha_output)
        ffn_output = self.ffn(skip_x_attention)
        ffn_output = self.dropout_ffn(ffn_output,training=training)
        encode_layer_out = self.layernorm2(skip_x_attention+ffn_output)

        return encode_layer_out

class Encoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,embedding_dim,num_heads,fully_connected_dim,input_vocab_size,maximum_position_encoding,dropout_rate=0.1,layernorm_eps=1e-6):
        super(Encoder,self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size,self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding,self.embedding_dim)
        self.enc_layers = [EncoderLayer(embedding_dim = self.embedding_dim, num_heads=num_heads,fully_connected_dim=fully_connected_dim,dropout_rate=dropout_rate,layernorm_eps=layernorm_eps) for _ in range(self.num_layers)]
        self.dropout = Dropout(dropout_rate)
    def call(self,x,training=None,mask=None):
        seq_length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim,tf.float32))
        x += self.pos_encoding[:,:seq_length,:]
        x = self.dropout(x,training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x,training=training,mask=mask)
        
        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,embedding_dim,num_heads,fully_connected_dim,dropout_rate=0.1,layernorm_eps=1e-6):
        super(DecoderLayer,self).__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads,key_dim=embedding_dim,dropout=dropout_rate)
        self.mha2 = MultiHeadAttention(num_heads=num_heads,key_dim=embedding_dim,dropout=dropout_rate)
        self.ffn = FullyConnected(embedding_dim=embedding_dim,fully_connected_dim=fully_connected_dim)
        self.layernorm1 = LayerNormalization(epsilon = layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon = layernorm_eps)
        self.layernorm3 = LayerNormalization(epsilon = layernorm_eps)
        self.dropout_ffn = Dropout(dropout_rate)
    def call(self,x,enc_output,training=None,look_ahead_mask=None,padding_mask=None):
        # Block 1
        mult_attn_out1, attn_weights_block1 = self.mha1(query=x, value=x, key=x, attention_mask=look_ahead_mask, return_attention_scores=True)
        Q1 = self.layernorm1(x+mult_attn_out1)
        # Block 2
        mult_attn_out2, attn_weights_block2 = self.mha2(query=Q1, value=enc_output, key=enc_output, attention_mask=padding_mask, return_attention_scores=True)
        mult_attn_out2 = self.layernorm2(Q1+mult_attn_out2)
        # Block 3
        ffn_output = self.ffn(mult_attn_out2)
        ffn_output = self.dropout_ffn(ffn_output,training=training)
        out3 = self.layernorm3(mult_attn_out2+ffn_output)

        return out3, attn_weights_block1, attn_weights_block2

class Decoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,embedding_dim,num_heads,fully_connected_dim,target_vocab_size,maximum_position_encoding,dropout_rate=0.1,layernorm_eps=1e-6):
        super(Decoder,self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(target_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)
        self.dec_layers = [DecoderLayer(embedding_dim=self.embedding_dim,num_heads=num_heads,fully_connected_dim=fully_connected_dim,dropout_rate=dropout_rate,layernorm_eps=layernorm_eps) for _ in range(self.num_layers)]
        self.dropout = Dropout(dropout_rate)
    def call(self,x,enc_output,training=None,look_ahead_mask=None,padding_mask=None):
        seq_length = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim,tf.float32))
        x += self.pos_encoding[:, :seq_length, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x,block1,block2 = self.dec_layers[i](x,enc_output,training=training,look_ahead_mask=look_ahead_mask,padding_mask=padding_mask)
            attention_weights['decoder_layer{}_block1_self_att'.format(i+1)]=block1
            attention_weights['decoder_layer{}_block2_decenc_att'.format(i+1)]=block2
        
        return x,attention_weights

class Transformer(tf.keras.Model):
    def __init__(self,num_layers,embedding_dim,num_heads,fully_connected_dim,input_vocab_size,target_vocab_size,max_positional_encoding_input,max_positional_encoding_target,dropout_rate=0.1,layernorm_eps=1e-6):
        super(Transformer,self).__init__()
        self.encoder = Encoder(num_layers=num_layers,embedding_dim=embedding_dim,num_heads=num_heads,fully_connected_dim=fully_connected_dim,input_vocab_size=input_vocab_size,maximum_position_encoding=max_positional_encoding_input,dropout_rate=dropout_rate,layernorm_eps=layernorm_eps)
        self.decoder = Decoder(num_layers=num_layers,embedding_dim=embedding_dim,num_heads=num_heads,fully_connected_dim=fully_connected_dim,target_vocab_size=target_vocab_size,maximum_position_encoding=max_positional_encoding_target,dropout_rate=dropout_rate,layernorm_eps=layernorm_eps)
        self.final_layer = Dense(target_vocab_size,activation='softmax')
    def call(self,input_sentence,output_sentence,training=None,enc_padding_mask=None,look_ahead_mask=None,dec_padding_mask=None):
        enc_output = self.encoder(input_sentence,training=training,mask=enc_padding_mask)
        dec_output, attention_weights = self.decoder(output_sentence,enc_output,training=training,look_ahead_mask=look_ahead_mask,padding_mask=dec_padding_mask)
        final_output = self.final_layer(dec_output)

        return final_output,attention_weights