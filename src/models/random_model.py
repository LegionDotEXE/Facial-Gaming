import tensorflow as tf
from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.optimizers import RMSprop, Adam

# Same path as transfered_model.py - we reuse it purely for its architecture
FACE_MODEL_PATH = 'results/basic_model_20_epochs_timestamp_1785527267.keras'


class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        # load the same architecture as transfered_model.py, trimmed the
        # same way (drop softmax + dropout) so both models are directly
        # comparable
        base_model = models.load_model(FACE_MODEL_PATH)
        truncated_base = Sequential(base_model.layers[:-2])
        truncated_base.build((None,) + input_shape)

        # randomize the weights instead of keeping the learned ones
        self._randomize_layers(truncated_base)

        # leave everything trainable - this is the "no transfer" control
        truncated_base.trainable = True

        self.model = Sequential([
            truncated_base,
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(categories_count, activation='softmax')
        ])

    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    @staticmethod
    def _randomize_layers(model):
        for layer in model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                new_kernel = layer.kernel_initializer(shape=layer.kernel.shape)
                layer.kernel.assign(new_kernel)
            if hasattr(layer, 'bias') and layer.bias is not None:
                new_bias = layer.bias_initializer(shape=layer.bias.shape)
                layer.bias.assign(new_bias)
