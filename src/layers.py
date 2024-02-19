import tensorflow as tf
from tensorflow.keras import layers


# TODO consider using dynamic convolutions https://arxiv.org/pdf/1912.03458.pdf
class ConvBlock(tf.keras.layers.Layer):
    """
    Convolutional block with Group Normalization and PReLU activation including residual connection.

    Pre-activation ResNet block to help convergence and improve acc. as proposed in the paper
    "Identity Mappings in Deep Residual Networks" by Kaiming He et al.
    https://arxiv.org/pdf/1603.05027.pdf

    Group normalization to make normalization independent of the batch size
    as proposed in the paper "Group Normalization" by Yuxin Wu et al.
    https://arxiv.org/abs/1803.08494

    PReLU activation to avoid the vanishing gradient problem and improve convergence
    as proposed in the paper "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" by Kaiming He et al.
    https://arxiv.org/abs/1502.01852

    Args:
        filters: Number of filters in the convolutional layers
        kernel_size: Size of the convolutional kernel
        groups: Number of groups for group normalization
        strides: Strides for the convolutional layers
        padding: Padding for the convolutional layers
    """
    def __init__(self, filters, kernel_size=3, groups=16, strides=1, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.GroupNormalization(groups=groups, axis=-1)
        self.act1 = layers.PReLU()
        self.conv1 = layers.Conv3D(filters, kernel_size=kernel_size, strides=strides, padding=padding)
        self.norm2 = layers.GroupNormalization(groups=groups, axis=-1)
        self.act2 = layers.PReLU()
        self.conv2 = layers.Conv3D(filters, kernel_size=kernel_size, strides=strides, padding=padding)
        self.add = layers.Add()

    def call(self, x):
        """
        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        skip = x
        x = self.norm1(x)
        x = sef.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = self.add([x, skip])
        return x


class BaseAttention(tf.keras.layers.Layer):
    """
    Base attention layer with multi-head attention, residual connection and layer normalization.
    Taken from the official TensorFlow documentation:
    https://www.tensorflow.org/text/tutorials/transformer

    Args:
        dropout: Dropout rate for the multi-head attention
        **kwargs: Additional keyword arguments for the multi-head attention
    """
    def __init__(self, dropout=0.1 **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(dropout, **kwargs)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()


class CrossAttention(BaseAttention):
    """
    Cross attention layer with multi-head attention, residual connection and layer normalization.
    Taken from the official TensorFlow documentation:
    https://www.tensorflow.org/text/tutorials/transformer

    Args:
        dropout: Dropout rate for the multi-head attention
        **kwargs: Additional keyword arguments for the multi-head attention
    """
    def call(self, x, context):
        """
        Args:
            x: Query tensor
            context: Key and value tensor

        Returns:
            Cross-attention output tensor
        """
        attn_output, attn_scores = self.mha(
                query=x,
                key=context,
                value=context,
                return_attention_scores=True)
        x = self.add([x, attn_output])
        return self.layernorm(x)


class GlobalSelfAttention(BaseAttention):
    """
    Global self-attention layer with multi-head attention, residual connection and layer normalization.
    Taken from the official TensorFlow documentation:
    https://www.tensorflow.org/text/tutorials/transformer

    Args:
        dropout: Dropout rate for the multi-head attention
        **kwargs: Additional keyword arguments for the multi-head attention
    """
    def call(self, x):
        """
        Args:
            x: Input tensor

        Returns:
            Global self-attention output tensor
        """
        attn_output = self.mha(
                query=x,
                value=x,
                key=x)
        x = self.add([x, attn_output])
        return self.layernorm(x)


class SeqPool(layers.Layer):
    """
    Sequence pooling layer with a single dense layer and no bias as proposed in the paper
    "Escaping the Big Data Paradigm with Compact Transformers" by Ali Hassani et al.
    https://arxiv.org/abs/2104.05704
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = layers.Dense(1, use_bias=False)

    def call(self, x):
        return self.pool(x)


class EncoderBlock(tf.keras.layers.Layer):
    """
    Encoder block with convolutional block, global self-attention and max pooling.

    Args:
        filters: Number of filters in the convolutional layers
        kernel_size: Size of the convolutional kernel
        groups: Number of groups for group normalization
    """
    def __init__(self, filters, kernel_size, groups, strides, padding, dropout, **kwargs):
        super().__init__(**kwargs)
        self.conv = ConvBlock(filters, kernel_size, groups) # on spatial axes
        self.gsa = GlobalSelfAttention(dropout=0.1) # on temporal axis
        self.pool = tf.keras.layers.MaxPooling3D()

    def call(self, x):
        x = self.conv(x)
        skip_connection = x
        x = self.gsa(x)
        x = self.pool(x)
        return x, skip_connection


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, groups, **kwargs):
        super().__init__(**kwargs)
        self.deconv = layers.Conv3DTranspose(filters, kernel_size=kernel_size, padding='same')
        self.cross_attention = CrossAttention()
        self.conv = ConvBlock(filters, kernel_size, groups)

    def call(self, x, skip):
        x = self.deconv(x)
        x = self.cross_attention(x, skip)
        x = self.conv(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    """

    As proposed in the paper "Attention is All You Need" by Ashish Vaswani et al.
    https://arxiv.org/abs/1706.03762

    GELU activation function as proposed in the paper "Gaussian Error Linear Units (GELUs)" by Dan Hendrycks et al.
    https://arxiv.org/abs/1606.08415
    """
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(dff, activation='gelu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(d_model, activation='gelu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        skip = x
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.add([skip, x])
        x = self.layer_norm(x)
        return x

# CCT uses 2 layer transformer encoder with 2 heads and 64 dims per head and 128 mlp_dim
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.gsa = GlobalSelfAttention(d_model, num_heads, mlp_dim, dropout=dropout)
        self.ff = FeedForward(d_model, mlp_dim, dropout_rate=dropout)


    def call(self, x):
        x = self.ln(x)
        x = self.gsa(x)
        x = self.ff(x)
        return x

