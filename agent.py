import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class Agent:
    def __init__(self, n_arms, gamma, lam):
        self.n_arms, self.gamma, self.lam = n_arms, gamma, lam
        self.actor, self.actor_opt = self.actor_net(), Adam(learning_rate=1e-4)
        self.critic, self.critic_opt = self.critic_net(), Adam(learning_rate=1e-4)

    def actor_net(self):
        return Sequential([Dense(self.n_arms), Dense(self.n_arms), Softmax()])

    def critic_net(self):
        return Sequential([Dense(self.n_arms), Dense(1)])

    def calc_policy(self, inputs):
        return self.actor(np.expand_dims(inputs, axis=0)).numpy()[0]

    def backprop(self, cur_input, next_inputs, arms, rewards):
        input_tensor, arm_tensor = np.expand_dims(cur_input, axis=0), np.expand_dims(arms, axis=1)
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            policy = tf.squeeze(self.actor(input_tensor, training=True))
            cur_value = tf.squeeze(self.critic(input_tensor, training=True))
            next_values = tf.squeeze(self.critic(next_inputs, training=True))
            temporal_diffs = rewards + self.gamma * next_values - cur_value
            log_probs = tf.math.log(tf.clip_by_value(policy, 0.0001, 0.999))
            action_diffs = tf.SparseTensor(arm_tensor, temporal_diffs, [len(policy)])
            actor_loss = tf.reduce_sum(-log_probs * tf.sparse.to_dense(action_diffs))
            critic_loss = tf.reduce_sum(temporal_diffs ** 2)
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
