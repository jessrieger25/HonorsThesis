import tensorflow as tf



# Step 1: define the placeholders for input and output
with tf.name_scope("data"):
  center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
  target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')
# Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
with tf.device('/cpu:0'):
  with tf.name_scope("embed"):
      # Step 2: define weights. In word2vec, it's actually the weights that we care about
      embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0),
           name='embed_matrix')
# Step 3 + 4: define the inference + the loss function
with tf.name_scope("loss"):
  # Step 3: define the inference
  embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
  # Step 4: construct variables for NCE loss
  nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],  stddev=1.0 / math.sqrt(EMBED_SIZE)),name='nce_weight')
  nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')
  # define loss function to be NCE loss function
  loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=target_words, inputs=embed, num_sampled=NUM_SAMPLED,
                                       num_classes=VOCAB_SIZE), name='loss')
  # Step 5: define optimizer
  optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)


with tf.Session() as sess:
 sess.run(tf.global_variables_initializer())
 average_loss = 0.0
 for index in xrange(NUM_TRAIN_STEPS):
 batch = batch_gen.next()
 loss_batch, _ = sess.run([loss, optimizer],
 feed_dict={center_words: batch[0], target_words: batch[1]})
 average_loss += loss_batch
 if (index + 1) % 2000 == 0:
 print('Average loss at step {}: {:5.1f}'.format(index + 1,
 average_loss / (index + 1)))