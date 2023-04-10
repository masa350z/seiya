# %%
from keras import layers
import tensorflow as tf
import numpy as np
import boatdata
from tqdm import tqdm
# %%


def scaled_dot_product_attention(q, k, v, mask=None):
    """アテンションの重みの計算
    q, k, vは最初の次元が一致していること
    k, vは最後から2番めの次元が一致していること
    マスクは型（パディングかルックアヘッドか）によって異なるshapeを持つが、
    加算の際にブロードキャスト可能であること
    引数：
     q: query shape == (..., seq_len_q, depth)
     k: key shape == (..., seq_len_k, depth)
     v: value shape == (..., seq_len_v, depth_v)
     mask: (..., seq_len_q, seq_len_k) にブロードキャスト可能な
      shapeを持つ浮動小数点テンソル。既定値はNone
    戻り値：
     出力、アテンションの重み
     """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # matmul_qkをスケール
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # マスクをスケール済みテンソルに加算
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax は最後の軸(seq_len_k)について
    # 合計が1となるように正規化
    # (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        layers.Dense(d_model)  # (batch_size, seq_len, d_model)
      ])


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))

    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    poisition: シーケンスの最大長
    d_model: 1シーケンスの次元数（単語ベクトルの次元数）
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 配列中の偶数インデックスにはsinを適用; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 配列中の奇数インデックスにはcosを適用; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """最後の次元を(num_heads, depth)に分割。
        結果をshapeが(batch_size, num_heads, seq_len, depth)となるようにリシェイプする。
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
           q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 max_sequence_len, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_sequence_len,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # 埋め込みと位置エンコーディングを合算する
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class NoEmbeddingEncoder(layers.Layer):
    def __init__(self, seq_len, num_layers, d_model, num_heads, dff, rate=0.1):
        super(NoEmbeddingEncoder, self).__init__()
        self.seq_len = seq_len

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(self.seq_len, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask):
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :self.seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class BoatEncoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 max_sequence_len, output_size, rate=0.1):
        super(BoatEncoder, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, max_sequence_len, rate)

        self.final_layer = layers.Dense(output_size)

    def call(self, inp, training=False, enc_padding_mask=None):

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(enc_output)

        return final_output


class PreinfoEncoder(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, output_size):
        super(PreinfoEncoder, self).__init__()

        self.encoder = NoEmbeddingEncoder(6, num_layers, 8, num_heads, dff)

        self.final_layer = layers.Dense(output_size)

    def call(self, inp, training=False, enc_padding_mask=None):

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(enc_output)

        return final_output


class OddsEncoder(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, output_size):
        super(OddsEncoder, self).__init__()

        self.encoder = NoEmbeddingEncoder(1, num_layers, 120, num_heads, dff)

        self.final_layer = layers.Dense(output_size)

    def call(self, inp, training=False, enc_padding_mask=None):

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(enc_output)

        return final_output

# %%
class Sanren120(boatdata.BoatDataset):
    """
    ラベルが3連単の組み合わせ120通りのデータセット
    (x, 120)次元のラベル
    """
    def __init__(self, n, ret_grade=True, sorted=True):
        super().__init__(n, ret_grade, sorted)
        self.odds = self.ret_sorted_odds()
        self.set_label()

    def set_label(self):
        label = np.zeros((len(self.ar_num), 120))
        th_ar = self.ret_sorted_th()[:, :3]

        for i in tqdm(range(len(th_ar))):
            temp_ar = th_ar[i]

            sanren = ''
            for nm in temp_ar:
                sanren += str(nm)

            label[i][self.sanren_dic[sanren]] = 1

        self.label = label

    def set_dataset(self, batch_size):
        mask = np.array([False, False, False, False, False, False, False, False,
                         False, True, True, True, True, True, True, True, True, True, True, True, True, False])
        inp = self.tokenized_inputs[:, mask]

        mn = np.min(np.where(inp == 100, 100000, inp))

        inp = inp - mn + 1
        inp = np.where(inp == -11500, 0, inp)

        field_mask = np.array([False, True, True, True, True, True, True, True,
                               False, False, False, False, False, False, False, False, False, False, False, False, False, False])
        inp_fe = self.tokenized_inputs[:, field_mask]
        mn = np.min(np.where(inp_fe == 100, 100000, inp_fe))
        inp_fe = inp_fe - mn

        pre_mask = np.array([True, True, True,  True, True, True,  True,  True])
        pre_info = self.pre_info[:, :, pre_mask]

        self.x_train, self.x_valid, self.x_test = boatdata.split_data(inp)
        self.field_train, self.field_valid, self.field_test = boatdata.split_data(inp_fe)
        self.pre_train, self.pre_valid, self.pre_test = boatdata.split_data(pre_info)
        self.odds_train, self.odds_valid, self.odds_test = boatdata.split_data(self.odds)
        self.y_train, self.y_valid, self.y_test = boatdata.split_data(self.label)

        x = tf.data.Dataset.from_tensor_slices((self.x_train)).batch(batch_size)
        field = tf.data.Dataset.from_tensor_slices((self.field_train)).batch(batch_size)
        pre = tf.data.Dataset.from_tensor_slices((self.pre_train)).batch(batch_size)
        odds = tf.data.Dataset.from_tensor_slices((self.odds_train)).batch(batch_size)
        train_y = tf.data.Dataset.from_tensor_slices((self.y_train)).batch(batch_size)
        self.train = tf.data.Dataset.zip((x, field, pre, odds))
        self.train = tf.data.Dataset.zip((self.train, train_y))

        x = tf.data.Dataset.from_tensor_slices((self.x_valid)).batch(batch_size)
        field = tf.data.Dataset.from_tensor_slices((self.field_valid)).batch(batch_size)
        pre = tf.data.Dataset.from_tensor_slices((self.pre_valid)).batch(batch_size)
        odds = tf.data.Dataset.from_tensor_slices((self.odds_valid)).batch(batch_size)
        valid_y = tf.data.Dataset.from_tensor_slices((self.y_valid)).batch(batch_size)
        self.valid = tf.data.Dataset.zip((x, field, pre, odds))
        self.valid = tf.data.Dataset.zip((self.valid, valid_y))

        x = tf.data.Dataset.from_tensor_slices((self.x_test)).batch(batch_size)
        field = tf.data.Dataset.from_tensor_slices((self.field_test)).batch(batch_size)
        pre = tf.data.Dataset.from_tensor_slices((self.pre_test)).batch(batch_size)
        odds = tf.data.Dataset.from_tensor_slices((self.odds_test)).batch(batch_size)
        test_y = tf.data.Dataset.from_tensor_slices((self.y_test)).batch(batch_size)
        self.test = tf.data.Dataset.zip((x, field, pre, odds))
        self.test = tf.data.Dataset.zip((self.test, test_y))

    def model_compile(self, learning_rate=False):
        if learning_rate:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = 'Adam'

        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def start_training(self, epochs, weight_name, k_freeze=3):
        best_val_loss = float('inf')
        k_freeze = 3
        freeze = k_freeze

        val_loss, val_acc = self.model.evaluate(self.valid)
        print(f"Initial valid loss: {val_loss}")

        # 学習を開始する
        for epoch in range(epochs):
            self.model.fit(self.train)
            val_loss, val_acc = self.model.evaluate(self.valid)

            # valid lossが減少した場合、重みを保存
            if val_loss < best_val_loss:
                freeze = 0
                best_val_loss = val_loss
                self.model.save_weights(weight_name)
                print(f"Epoch {epoch + 1}: Valid loss decreased to {val_loss}, saving weights.")

            # valid lossが減少しなかった場合、保存しておいた最良の重みをロード
            else:
                if freeze == 0:
                    self.model.load_weights(weight_name)
                    self.model_weights_random_init()
                    freeze = k_freeze
                    print(f"Epoch {epoch + 1}: Valid loss did not decrease, loading weights.")
                else:
                    print(f"Epoch {epoch + 1}: Valid loss did not decrease.")

            freeze = freeze - 1 if freeze > 0 else freeze

            print('')


# %%
bt = Sanren120(0.2)
bt.set_dataset(batch_size=360)
# %%


class BoatTransformer(tf.keras.Model):
    """
    num_layers: エンコードーを何回通すか
    d_model: 1シーケンスの次元数（単語ベクトルの次元数）
    num_heads: 入力の分割数 d_modelを割り切れる値
    dff: フィードフォワードNW の次元数
    input_vocab_size: 単語数
    max_sequence_len: シーケンスの最大長
    output_size: 出力層のユニット数
    """
    def __init__(self):
        super(BoatTransformer, self).__init__()
        num_layers = 6
        self.vect_len = 1024
        diff = 2048
        self.output_size = 1024

        self.racers_encoder = BoatEncoder(num_layers=num_layers,
                                          d_model=self.vect_len,
                                          num_heads=8,
                                          dff=diff,
                                          input_vocab_size=1382,
                                          max_sequence_len=12,
                                          output_size=self.output_size)

        self.field_encoder = BoatEncoder(num_layers=num_layers,
                                         d_model=self.vect_len,
                                         num_heads=8,
                                         dff=diff,
                                         input_vocab_size=1078,
                                         max_sequence_len=7,
                                         output_size=self.output_size)

        self.preinfo_encoder = PreinfoEncoder(num_layers=num_layers,
                                              num_heads=2,
                                              dff=diff,
                                              output_size=self.output_size)

        self.odds_encoder = OddsEncoder(num_layers=num_layers,
                                        num_heads=1,
                                        dff=diff,
                                        output_size=self.output_size)        

        self.senshu01 = layers.Dense(self.output_size, activation='relu')
        self.senshu02 = layers.Dense(self.output_size, activation='relu')
        self.senshu03 = layers.Dense(self.output_size, activation='relu')
        self.senshu04 = layers.Dense(self.output_size, activation='relu')
        self.senshu05 = layers.Dense(self.output_size, activation='relu')
        self.senshu06 = layers.Dense(self.output_size, activation='relu')

        self.layer01 = layers.Dense(2048, activation='relu')
        self.layer02 = layers.Dense(1024, activation='relu')

        self.output_layer = layers.Dense(120, activation='softmax')

    def call(self, inputs):
        x, field, pre, odds = inputs

        odds = tf.expand_dims(odds, 1)

        x = self.racers_encoder(x)
        field = self.field_encoder(field)
        pre = self.preinfo_encoder(pre)
        odds = self.odds_encoder(odds)[:, 0]

        x01 = tf.reshape(x[:, 0:2], (-1, 2*self.output_size))
        x02 = tf.reshape(x[:, 2:4], (-1, 2*self.output_size))
        x03 = tf.reshape(x[:, 4:6], (-1, 2*self.output_size))
        x04 = tf.reshape(x[:, 6:8], (-1, 2*self.output_size))
        x05 = tf.reshape(x[:, 8:10], (-1, 2*self.output_size))
        x06 = tf.reshape(x[:, 10:12], (-1, 2*self.output_size))

        x01 = self.senshu01(layers.concatenate([x01, pre[:, 0]]))
        x02 = self.senshu02(layers.concatenate([x02, pre[:, 1]]))
        x03 = self.senshu03(layers.concatenate([x03, pre[:, 2]]))
        x04 = self.senshu04(layers.concatenate([x04, pre[:, 3]]))
        x05 = self.senshu05(layers.concatenate([x05, pre[:, 4]]))
        x06 = self.senshu06(layers.concatenate([x06, pre[:, 5]]))

        field = tf.reshape(field, (-1, 7*self.output_size))

        x = self.layer01(layers.concatenate([field, odds, x01, x02, x03, x04, x05, x06]))

        x = self.layer02(x)
        x = self.output_layer(x)

        return x


# %%
bt.model = BoatTransformer()
bt.model_compile()
# %%
bt.start_training(epochs=100, weight_name='datas/sanren120/boat_transformer', k_freeze=1)
# %%
