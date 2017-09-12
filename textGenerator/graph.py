import tensorflow as tf
import numpy as np
import reader


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def build_graph(
        num_classes,
        cell_type=None,
        state_size=100,
        batch_size=32,
        num_steps=200,
        num_layers=3,
        build_with_dropout=False,
        learning_rate=1e-4,
        learning_rate_decay = 0.9):
    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
    global_step = tf.Variable(0,trainable=False)
    add_step = tf.assign_add(global_step,1)
    lr = tf.train.exponential_decay(learning_rate, global_step,decay_steps=40,decay_rate=learning_rate_decay)
    dropout = tf.constant(1.0)

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size],
                                 initializer=tf.random_normal_initializer())

    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    if cell_type == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(state_size)
    elif cell_type == 'LSTM':
        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    else:
        cell = tf.nn.rnn_cell.BasicRNNCell(state_size)

    if build_with_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)

    if cell_type == 'LSTM' or cell_type == 'LN_LSTM':
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    else:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    if build_with_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)

    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    # reshape rnn_outputs and y
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_reshaped, logits=logits))
    train_step = tf.train.AdamOptimizer(lr).minimize(total_loss)

    return dict(
        x=x,
        y=y,
        init_state=init_state,
        final_state=final_state,
        total_loss=total_loss,
        train_step=train_step,
        preds=predictions,
        saver=tf.train.Saver(),
        add_step=add_step
    )


def train_network(g, epochs, verbose=True, save=False):
    # tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        training_losses = []
        for idx, epoch in enumerate(epochs):
            training_loss = 0
            steps = 0
            training_state = None
            X = epoch[0]
            Y = epoch[1]
            steps += 1

            feed_dict = {g['x']: X, g['y']: Y}
            if training_state is not None:
                feed_dict[g['init_state']] = training_state
            try:
                training_loss_, training_state, _,_ = sess.run([g['total_loss'],
                                                              g['final_state'],
                                                              g['train_step'],
                                                              g['add_step']],
                                                              feed_dict)
            except ValueError:
                continue
            training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss / steps)
            training_losses.append(training_loss / steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses


def generate_characters(g, checkpoint, num_chars, reader, prompt='A', pick_top_chars=None):
    """ Accepts a current character, initial state"""
    char_id, id_char = reader.to_ids()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        g['saver'].restore(sess, checkpoint)

        state = None
        current_char = char_id[prompt]
        chars = [current_char]

        for i in range(num_chars):
            if state is not None:
                feed_dict = {g['x']: [[current_char]], g['init_state']: state}
            else:
                feed_dict = {g['x']: [[current_char]]}

            preds, state = sess.run([g['preds'], g['final_state']], feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0
                p = p / np.sum(p)
                current_char = np.random.choice(reader.len, 1, p=p)[0]
            else:
                current_char = np.random.choice(reader.len, 1, p=np.squeeze(preds))[0]

            chars.append(current_char)

    chars = [id_char[c] for c in chars]
    print("".join(chars))
    return ("".join(chars))
