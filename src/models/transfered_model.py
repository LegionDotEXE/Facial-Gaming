from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.optimizers import RMSprop, Adam

# Path to your best facial recognition model from Section 6
FACE_MODEL_PATH = 'results/basic_model_20_epochs_timestamp_1785527267.keras'


class TransferedModel(Model):
    def _define_model(self, input_shape, categories_count):
        # load your trained facial recognition model
        base_model = models.load_model(FACE_MODEL_PATH)

        # eliminate the final softmax layer AND the trailing Dropout layer.
        # Dropout has no weights, so freezing the base doesn't stop it from
        # randomly zeroing features on every training step - that injects
        # noise into what's supposed to be a stable, frozen feature vector.
        # Cutting it keeps the frozen output deterministic.
        truncated_base = Sequential(base_model.layers[:-2])
        truncated_base.build((None,) + input_shape)

        # freeze all parameters in the remainder so they cannot change
        # through further learning
        truncated_base.trainable = False

        # bolt on new fully connected layers + a new softmax for the
        # new (2-class) problem, and train only these new layers
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
