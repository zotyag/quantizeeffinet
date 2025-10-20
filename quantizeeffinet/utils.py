import tensorflow as tf

def make_model(num_classes: int, base_model: str = 'EfficientNetB3') -> tf.keras.Model:
    if base_model == 'EfficientNetB3':
        base = tf.keras.applications.EfficientNetB3(weights=None, include_top=False, input_shape=(224, 224, 3))
    elif base_model == 'EfficientNetB5':
        base = tf.keras.applications.EfficientNetB5(weights=None, include_top=False, input_shape=(224, 224, 3))
    elif base_model == 'EfficientNetB6':
        base = tf.keras.applications.EfficientNetB6(weights=None, include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError(f"Unsupported base_model: {base_model}")
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model