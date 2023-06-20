from keras.initializers import he_uniform
from keras.activations import gelu
from keras import layers
import tensorflow as tf
import numpy as np


def scaled_dot_product_attention(q, k, v):
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

    # softmax は最後の軸(seq_len_k)について
    # 合計が1となるように正規化
    # (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(vector_dims, inner_dims):
    return tf.keras.Sequential([
        layers.Dense(inner_dims,
                     kernel_initializer=he_uniform(),
                     activation=gelu),  # (batch_size, seq_len, dff)
        layers.Dense(vector_dims)  # (batch_size, seq_len, d_model)
      ])


def get_angles(pos, i, vector_dims):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(vector_dims))

    return pos*angle_rates


def positional_encoding(position, vector_dims, trainable=False):
    """
    poisition: シーケンスの最大長
    d_model: 1シーケンスの次元数（単語ベクトルの次元数）
    """
    if trainable:
        pos_encoding = tf.Variable(tf.random.normal(shape=(1, position, vector_dims)),
                                   trainable=True, dtype=tf.float32)

        return pos_encoding
    else:
        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(vector_dims)[np.newaxis, :],
                                vector_dims)

        # 配列中の偶数インデックスにはsinを適用; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # 配列中の奇数インデックスにはcosを適用; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(layers.Layer):
    def __init__(self, vector_dims, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.vector_dims = vector_dims

        assert vector_dims % self.num_heads == 0

        self.depth = vector_dims // self.num_heads

        self.wq = layers.Dense(vector_dims)
        self.wk = layers.Dense(vector_dims)
        self.wv = layers.Dense(vector_dims)

        self.dense = layers.Dense(vector_dims)

    def split_heads(self, x, batch_size):
        """最後の次元を(num_heads, depth)に分割。
        結果をshapeが(batch_size, num_heads, seq_len, depth)となるようにリシェイプする。
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
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
           q, k, v)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.vector_dims))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class OutputLayer(layers.Layer):
    def __init__(self, vector_dims, num_heads, inner_dims):
        super(OutputLayer, self).__init__()

        self.mha = MultiHeadAttention(vector_dims, num_heads)
        self.ffn = point_wise_feed_forward_network(vector_dims, inner_dims)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):

        attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class TransformerBase(layers.Layer):
    def __init__(self, num_layer_loops, vector_dims,
                 num_heads, inner_dims):
        super(TransformerBase, self).__init__()

        self.num_layer_loops = num_layer_loops
        self.vector_dims = vector_dims
        self.num_heads = num_heads
        self.inner_dims = inner_dims

        self.cls_embedding = layers.Embedding(1, vector_dims)

        self.enc_layers = [OutputLayer(vector_dims, num_heads, inner_dims)
                           for _ in range(num_layer_loops)]

    def add_cls(self, x, batch_size):
        cls = self.cls_embedding(0)
        cls = tf.reshape(cls, (1, 1, -1))
        cls = tf.tile(cls, [batch_size, 1, 1])
        x = tf.concat([cls, x], axis=1)

        return x


class OutputDense(layers.Layer):
    def __init__(self, feature_dims):
        super(OutputDense, self).__init__()

        self.dense01 = layers.Dense(feature_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)
        self.dense02 = layers.Dense(feature_dims,
                                    activation=gelu,
                                    kernel_initializer=he_uniform)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x_ = self.dense01(x)
        x = self.layernorm1(x_ + x)
        x_ = self.dense02(x)
        x = self.layernorm2(x_ + x)

        return x


class MaskedEmbedding(layers.Layer):
    def __init__(self, sequence_len, vector_dims):
        super(MaskedEmbedding, self).__init__()

        self.embedding = layers.Embedding(sequence_len, vector_dims)

    def call(self, x):
        mask = tf.expand_dims(x > 0, 3)
        x = self.embedding(x)

        return x*tf.cast(mask, tf.float32)


class RacerTransformer(TransformerBase):
    def __init__(self, num_layer_loops, vector_dims,
                 num_heads, inner_dims):
        super(RacerTransformer, self).__init__(num_layer_loops, vector_dims,
                                               num_heads, inner_dims)

        self.racer_embedding = layers.Embedding(10000, vector_dims)
        self.grade_embedding = layers.Embedding(4, vector_dims)
        self.incose_embedding = layers.Embedding(6, vector_dims)

    def call(self, x):
        racer, grade, incose = x
        batch_size = racer.shape[0]

        racer = self.racer_embedding(racer)
        grade = self.grade_embedding(grade)
        pos_encoding = self.incose_embedding(incose)

        x = racer + grade + pos_encoding

        x = self.add_cls(x, batch_size)

        for i in range(self.num_layer_loops):
            x += self.enc_layers[i](x)

        return x, pos_encoding


class F_L_aveST_Encoder(layers.Layer):
    def __init__(self, feature_dims):
        super(F_L_aveST_Encoder, self).__init__()

        self.flying_embedding = layers.Embedding(10, feature_dims)
        self.latestart_embedding = layers.Embedding(10, feature_dims)
        self.avest_encoder = layers.Dense(feature_dims)

        self.output_dense = OutputDense(feature_dims)

    def call(self, x):
        flying, latestart, avest = x

        flying = self.flying_embedding(flying)
        latestart = self.latestart_embedding(latestart)
        avest = self.avest_encoder(tf.expand_dims(avest, 2))

        return self.output_dense(flying + latestart + avest)


class RacerWinningRateEncoder(layers.Layer):
    def __init__(self, feature_dims):
        super(RacerWinningRateEncoder, self).__init__()

        self.zenkoku_encoder = layers.Dense(feature_dims)
        self.touchi_encoder = layers.Dense(feature_dims)

        self.output_dense = OutputDense(feature_dims)

    def call(self, x):
        zenkoku, touchi = x

        zenkoku = self.zenkoku_encoder(zenkoku)
        touchi = self.touchi_encoder(touchi)

        return self.output_dense(zenkoku + touchi)


class MotorBoatWinningRateEncoder(layers.Layer):
    def __init__(self, feature_dims):
        super(MotorBoatWinningRateEncoder, self).__init__()

        self.motor_encoder = layers.Dense(feature_dims)
        self.boat_encoder = layers.Dense(feature_dims)

        self.output_dense = OutputDense(feature_dims)

    def call(self, x):
        motor, boat = x

        motor = self.motor_encoder(motor)
        boat = self.boat_encoder(boat)

        return self.output_dense(motor + boat)


class CurrentInfoTransformer(TransformerBase):
    def __init__(self, num_layer_loops, vector_dims, num_heads, inner_dims):
        super(CurrentInfoTransformer, self).__init__(num_layer_loops,
                                                     vector_dims,
                                                     num_heads,
                                                     inner_dims)

        self.no_embedding = MaskedEmbedding(7, vector_dims)
        self.cose_embedding = MaskedEmbedding(7, vector_dims)
        self.result_embedding = MaskedEmbedding(7, vector_dims)
        self.start_encoder = layers.Dense(vector_dims)

        self.positional = tf.Variable(tf.random.normal(shape=(1, 14, vector_dims)),
                                      trainable=True)

    def call(self, x):
        no, cose, result, start = x
        batch_size = no.shape[0]

        no = self.no_embedding(no)
        cose = self.cose_embedding(cose)
        result = self.result_embedding(result)
        start = self.start_encoder(np.expand_dims(start, 3))

        current_info = no + cose + result + start

        x = tf.reshape(current_info, (batch_size*6, 14, self.vector_dims))
        x = x + self.positional

        x = self.add_cls(x, batch_size*6)

        for i in range(self.num_layer_loops):
            x += self.enc_layers[i](x)

        return tf.reshape(x, (batch_size, 6, 15, self.vector_dims))


class StartTenjiEncoder(layers.Layer):
    def __init__(self, feature_dims):
        super(StartTenjiEncoder, self).__init__()

        self.start_encoder = layers.Dense(feature_dims)
        self.tenji_encoder = layers.Dense(feature_dims)

        self.output_dense = OutputDense(feature_dims)

    def call(self, x):
        start, tenji = x

        mx = tf.reduce_max(start, axis=1, keepdims=True)
        mn = tf.reduce_min(start, axis=1, keepdims=True)

        start = (start - mn) / (mx - mn) + 1

        start = self.start_encoder(tf.expand_dims(start, 2))
        tenji = self.tenji_encoder(tf.expand_dims(tenji, 2))

        return self.output_dense(start + tenji)


class ComputerPredictionTransformer(TransformerBase):
    def __init__(self, num_layer_loops, vector_dims, num_heads, inner_dims):
        super(ComputerPredictionTransformer, self).__init__(num_layer_loops,
                                                            vector_dims,
                                                            num_heads,
                                                            inner_dims)

        self.prediction_embedding = layers.Embedding(6, vector_dims)
        self.confidence_embedding = layers.Embedding(5, vector_dims)
        self.positional = tf.Variable(tf.random.normal(shape=(1, 26, vector_dims)),
                                      trainable=True)

    def call(self, x):
        prediction, confidence = x
        batch_size = prediction.shape[0]

        prediction = self.prediction_embedding(prediction)
        confidence = self.confidence_embedding(confidence)
        confidence = tf.expand_dims(confidence, 1)

        x = prediction + confidence
        x = x + self.positional

        x = self.add_cls(x, batch_size)

        for i in range(self.num_layer_loops):
            x += self.enc_layers[i](x)

        return x


class PredictionMarkEncoder(layers.Layer):
    def __init__(self, feature_dims):
        super(PredictionMarkEncoder, self).__init__()

        self.mark_embedding = layers.Embedding(5, feature_dims)
        self.output_dense = OutputDense(feature_dims)

    def call(self, x):
        x = self.mark_embedding(x)

        return self.output_dense(x)


class FieldEncoder(layers.Layer):
    def __init__(self, feature_dims):
        super(FieldEncoder, self).__init__()

        self.field_embedding = layers.Embedding(24, feature_dims)
        self.wether_embedding = layers.Embedding(6, feature_dims)
        self.wind_embedding = layers.Embedding(17, feature_dims)

        self.tempreture_encoder = layers.Dense(feature_dims)
        self.wind_speed_encoder = layers.Dense(feature_dims)
        self.water_tempreture_encoder = layers.Dense(feature_dims)
        self.water_hight_encoder = layers.Dense(feature_dims)

        self.output_dense = OutputDense(feature_dims)

    def call(self, x):
        field, wether, wind, temp, wind_speed, water_temp, water_hight = x

        field = self.field_embedding(field)
        wether = self.wether_embedding(wether)
        wind = self.wind_embedding(wind)
        temp = self.tempreture_encoder(tf.expand_dims(temp, 1))
        wind_speed = self.wind_speed_encoder(tf.expand_dims(wind_speed, 1))
        water_temp = self.water_tempreture_encoder(tf.expand_dims(water_temp, 1))
        water_hight = self.water_hight_encoder(tf.expand_dims(water_hight, 1))

        x = field + wether + wind + temp + wind_speed + water_temp + water_hight

        return self.output_dense(x)


class OddsTransformer(TransformerBase):
    def __init__(self, num_layer_loops, vector_dims, num_heads, inner_dims, seq_len=120):
        super(OddsTransformer, self).__init__(num_layer_loops,
                                              vector_dims,
                                              num_heads,
                                              inner_dims)

        self.odds_encoder = layers.Dense(vector_dims)

        self.positional = tf.Variable(tf.random.normal(shape=(1, seq_len, vector_dims)),
                                      trainable=True)

    def call(self, x):
        batch_size = x.shape[0]
        x = self.odds_encoder(tf.expand_dims(x, 2))
        x = x + self.positional

        x = self.add_cls(x, batch_size)

        for i in range(self.num_layer_loops):
            x += self.enc_layers[i](x)

        return x
