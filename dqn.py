



learning_rate = 0.0001
input_height = 34
input_width = 46
input_channels = 1
conv_n_maps = [32, 32, 32]
conv_kernel_strides = [(3,3), (3,3), (3,3)]
conv_strides = [1,1,1]
conv_padding = ["Same"] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 34*46*32
n_hidden = 64
hidden_activation = tf.nn.relu
n_ouputs = 4
initializer = tf.contrib.layers.variance_scaling_initializer()


def q_network(X_state, name):

	prev_layer = X_state
	conv_layers = []
	with tf.variable_scope(name) as scope:
		for n_maps, kernel_size, stride, padding, activation in zip(conv_n_maps, conv_kernel_strides, conv_strides, conv_padding, conv_activation):
			prev_layer = tf.layers.conv2d(prev_layer, filters=n_maps, kernel_size=kernel_size, 
										  strides=stride, padding=padding, activation=activation, kernel_initializer=initializer)
		last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
		hidden = tf.layers.dense(last_conv_layer_flat, n_hidden, activation=hidden_activation, kernel_initializer=initializer)
		outputs = tf.layers.dense(hidden, n_ouputs, kernel_initializer=initializer)
	trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
	trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
	return outputs, trainable_vars_by_name

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
actor_q_values, actor_vars = q_network(X_state, name="q_networks/actor")
critic_q_values, critic_vars = q_network(X_state, name="q_networks/critic")

copy_ops = [actor_var.assign(critic_vars[var_name]) for var_name, actor_name in actor_vars.items()]
copy_critic_to_actor = tf.group(*copy_ops)

X_action = tf.placeholder(tf.int32, shape=[None])
q_value = tf.reduce_sum(critic_q_values * tf.one_hot(X_action, n_ouputs), axis=1, keep_dims=True)

y = tf.placeholder(tf.float32, shape=[None, 1])
cost = tf.reduce_mean(tf.square(y - q_value))

global_step = tf.Variable(0, trainable=False, name="global_step")
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cost, global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

replay_memory_size = 10000
replay_memory = deque([], maxlen=replay_memory_size)

def sample_memories(batch_size):
	indices = np.random.permutation(len(replay_memory))[:batch_size]
	cols = [[],[],[],[],[]]
	for idx in indices:
		memory = replay_memory[idx]
		for col, value in zip(cols, memory):
			col.append(value)
	cols = [np.array(col) for col in cols]
	return (cols[0], cols[1], cols[2].reshape(-1,1), cols[3], cols[4].reshape(-1,1))

eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 50000

def epsilon_greedy(q_values, step):
	epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
	if np.random.rand() < epsilon:
		return np.random.randint(n_ouputs)
	else:
		return np.argmax(q_values)


n_steps = 100000
training_start = 1000
training_interval = 3
save_steps = 50
copy_steps = 25
discount_rate = 0.95
batch_size = 50
iteration = 0
checkpoint_path = "model/dqg.ckpt"
done = True

with tf.Session() as sess:

	if os.path.isfile(checkpoint_path):
		saver.restore(sess, checkpoint_path)
	else:
		sess.run(init)

	while True:
		step = global_step.eval()
		if step >= n_steps:
			break

		iteration += 1

		if done:

			done = False

			# restart game

		q_values = actor_q_values.eval(feed_dict={X_state: [state]})
		action = epsilon_greedy(q_values, step)

		# if alive reward = 1 else -1, if dead done = True
		next_state = #get screen

		replay_memory.append((state, action, reward, next_state, 1.0 - done))
		state = next_state

		if iteration < training_start or iteration % training_interval != 0:
			continue

		X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))
		next_q_values = actor_q_values.eval(feed_dict={X_state: X_next_state_val})
		max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
		y_val = rewards + continues * discount_rate * max_next_q_values
		training_op.run(feed_dict={X_state: X_state_val, X_action: X_action_val, y: y_val})

		if step % copy_steps == 0:
			copy_critic_to_actor.run()
		if step % save_steps == 0:
			saver.save(sess, checkpoint_path)