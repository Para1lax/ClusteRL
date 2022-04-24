import numpy as np
from numpy.random import choice
import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax, LayerNormalization
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adadelta


class Agent:
    def __init__(self, n_arms, gamma, lam):
        self.n_arms, self.gamma, self.lam = n_arms, gamma, lam
        self.actor, self.actor_opt = self.actor_net(), Adadelta(learning_rate=1e-3)
        self.critic, self.critic_opt = self.critic_net(), Adam(learning_rate=1e-3)

    def actor_net(self):
        return Sequential([
            Dense(self.n_arms, kernel_regularizer=L1L2(1e-4, 1e-3), bias_regularizer=L1L2(1e-4, 1e-3)),
            Dense(self.n_arms), Softmax()
        ])

    def critic_net(self):
        return Sequential([
            Dense(self.n_arms, kernel_regularizer=L1L2(1e-4, 1e-3), bias_regularizer=L1L2(1e-4, 1e-3)), Dense(1)
        ])

    def sample_action(self, inputs):
        policies = self.actor(np.expand_dims(inputs, 0))
        return choice(self.n_arms, p=policies.numpy()[0])

    def backprop(self, cur_inputs, next_inputs, arms, rewards):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            policies = self.actor(cur_inputs, training=True)
            cur_value = tf.squeeze(self.critic(cur_inputs, training=True))
            next_value = tf.squeeze(self.critic(next_inputs, training=True))
            temporal_diffs = rewards + self.gamma * next_value - cur_value
            log_probs = tf.math.log(tf.clip_by_value(policies, 0.0001, 0.999))
            action_diffs = tf.transpose(tf.one_hot(arms, self.n_arms)) * temporal_diffs
            actor_loss = tf.reduce_mean(-log_probs * tf.transpose(action_diffs))
            critic_loss = tf.reduce_mean(temporal_diffs ** 2)
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
