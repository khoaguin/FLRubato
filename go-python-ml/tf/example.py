import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        8, 
        (3, 3),
        strides=(2, 2),
        padding="valid",
        input_shape=(28, 28, 1),
        activation=tf.nn.relu,
        name="inputs"
    ),  # 14x14x8
    tf.keras.layers.Conv2D(
        16, (3, 3), strides=(2, 2), padding="valid", activation=tf.nn.relu
    ),  # 7x716
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, name="logits")  # linear
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Create some dummy data to build the model
dummy_data = tf.random.normal([1, 28, 28, 1])
_ = model(dummy_data)  # This builds the model

# Now save the model
tf.saved_model.save(model, "output/keras")