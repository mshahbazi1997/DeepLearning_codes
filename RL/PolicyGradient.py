import tensorflow as tf
import gym
import numpy as np

class Agent:
    def __init__(self, learning_rate):
        # Build the network to predict the correct action
        tf.reset_default_graph()
        input_dimension = 4
        hidden_dimension = 20
        self.input = tf.placeholder(dtype=tf.float32, shape=[1, input_dimension], name='X')
        
        w1 = tf.get_variable('w1',[input_dimension,hidden_dimension],initializer=tf.glorot_normal_initializer())
        b1 = tf.get_variable('b1',[hidden_dimension,],initializer=tf.initializers.zeros())
        hidden_layer = tf.nn.leaky_relu(tf.matmul(self.input,w1) + b1)
        w2 = tf.get_variable('w2',[hidden_dimension,2],initializer=tf.glorot_normal_initializer())
        b2 = tf.get_variable('b2',[2,],initializer=tf.initializers.zeros())
        logits = tf.matmul(hidden_layer,w2) + b2
        self.logits = logits

        # Sample an action according to network's output
        # use tf.multinomial and sample one action from network's output
        self.action = tf.multinomial(tf.log(tf.nn.softmax(logits)), num_samples = 1, name='action')

        # Optimization according to policy gradient algorithm
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.action,depth=2), logits=logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # use one of tensorflow optimizers
        
        
        grads_vars = self.optimizer.compute_gradients(
            cross_entropy)  # gradient of current action w.r.t. network's variables
        self.gradients = [grad for grad, var in grads_vars]

        # get rewards from the environment and evaluate rewarded gradients
        #  and feed it to agent and then call train operation
        self.rewarded_grads_placeholders_list = []
        rewarded_grads_and_vars = []
        for grad, var in grads_vars:
            rewarded_grad_placeholder = tf.placeholder(dtype=tf.float32, shape=grad.shape)
            self.rewarded_grads_placeholders_list.append(rewarded_grad_placeholder)
            rewarded_grads_and_vars.append((rewarded_grad_placeholder, var))

        self.train_operation = self.optimizer.apply_gradients(rewarded_grads_and_vars)

        self.saver = tf.train.Saver()

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        self.ses = tf.Session(config=config)
        self.ses.run(tf.global_variables_initializer())

    def get_action_and_gradients(self, obs):
        # compute network's action and gradients given the observations
        action, gradients = self.ses.run([self.action, self.gradients], feed_dict={self.input: np.asarray(obs)[np.newaxis]})
        action = action[0][0]
        return action, gradients

    def train(self, rewarded_gradients):
        feed_dict = {variable: rewarded_gradients[i] for i, variable in enumerate(self.rewarded_grads_placeholders_list)}
        # feed gradients into the placeholder and call train operation
        self.ses.run(self.train_operation, feed_dict = feed_dict)

    def save(self):
        self.saver.save(self.ses, "SavedModel/")

    def load(self):
        self.saver.restore(self.ses, "SavedModel/")
        
        
epochs = 110
max_steps_per_game = 1000
games_per_epoch = 20
discount_factor = 0.95
learning_rate = 0.01



agent = Agent(learning_rate)
game = gym.make("CartPole-v0").env

for epoch in range(epochs):

    epoch_rewards = []
    epoch_gradients = []
    epoch_average_reward = 0
    for episode in range(games_per_epoch):
        obs = game.reset()
        step = 0
        single_episode_rewards = []
        single_episode_gradients = []
        game_over = False
        while not game_over and step < max_steps_per_game:
            step += 1
            #image = game.render(mode='rgb_array') # Call this to render game and show visual
            action, gradients = agent.get_action_and_gradients(obs)
            obs, reward, game_over, info = game.step(action)
            single_episode_rewards.append(reward)
            single_episode_gradients.append(gradients)

        epoch_rewards.append(single_episode_rewards)
        epoch_gradients.append(single_episode_gradients)
        epoch_average_reward += sum(single_episode_rewards)

    epoch_average_reward /= games_per_epoch
    print("Epoch = {}, Average reward = {}".format(epoch, epoch_average_reward))

    # maximum length of game in one epoch
    length = []
    for reward in epoch_rewards:
        length.append(len(reward))
    max_len = max(length)


    epoch_normalized_rewards = []

    for i in range(len(length)):
        normalized_rewards = []
        for ii in range(length[i]):
            normalized_rewards.append((1-discount_factor**(ii+1))/(1-discount_factor)) # geometric sequence
        epoch_normalized_rewards.append(normalized_rewards)

    b = 0 # base line
    for i in range(len(epoch_rewards)):

        b+= (1-discount_factor**(length[i]+1))/(1-discount_factor)
    b = b/len(length)

    #normalized_rewards = np.asarray(normalized_rewards) - b 
    for i in range(len(length)):
        epoch_normalized_rewards[i] = np.asarray(epoch_normalized_rewards[i])-b

    dw1 = np.zeros_like(epoch_gradients[0][0][0])
    db1 = np.zeros_like(epoch_gradients[0][0][1])
    dw2 = np.zeros_like(epoch_gradients[0][0][2])
    db2 = np.zeros_like(epoch_gradients[0][0][3])


    N = len(epoch_gradients)
    for e in range(len(epoch_gradients)):
        all_grads = epoch_gradients[e]
        for t in range(len(all_grads)):
            grads = all_grads[t]

            dw1 += epoch_normalized_rewards[e][len(epoch_normalized_rewards[e])-t-1]*grads[0]
            db1 += epoch_normalized_rewards[e][len(epoch_normalized_rewards[e])-t-1]*grads[1]
            dw2 += epoch_normalized_rewards[e][len(epoch_normalized_rewards[e])-t-1]*grads[2]
            db2 += epoch_normalized_rewards[e][len(epoch_normalized_rewards[e])-t-1]*grads[3]
    dw1 = dw1/N
    db1 = db1/N
    dw2 = dw2/N
    db2 = db2/N

    mean_rewarded_gradients = [dw1, db1, dw2, db2]

    agent.train(mean_rewarded_gradients)
agent.save()
game.close()

# Run this part after training the network
# agent = Agent(0)
# game = gym.make("CartPole-v0").env
# agent.load()
# score = 0
# for i in range(10):
#     obs = game.reset()
#     game_over = False
#     while not game_over:
#         score += 1
#         image = game.render(mode='rgb_array')  # Call this to render game and show visual
#         action, _ = agent.get_action_and_gradients(obs)
#         obs, reward, game_over, info = game.step(action)
#
# print("Average Score = ", score / 10)
