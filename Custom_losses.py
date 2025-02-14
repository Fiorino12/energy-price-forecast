import sys
from typing import List
import numpy as np
import tensorflow as tf
import keras
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
"""
Set of tensorflow losses that can be utilized to train models
- PinballLoss : Average Pinball loss for QR method
- PinballLoss_Normal: Average Pinball Loss for Normal method
- WinklerLoss : Average Winkler loss
- CRPSLoss: CRPS Loss  
"""

class PinballLoss(keras.losses.Loss):
    def __init__(self, quantiles: List, name="pinball_loss"):
        super().__init__(name=name)
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        loss = []
        for i, q in enumerate(self.quantiles):
            error = tf.subtract(y_true, y_pred[:, :, i])
            loss_q = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
            loss.append(loss_q)
        L = tf.convert_to_tensor(loss)
        total_loss = tf.reduce_mean(L)
        return total_loss

class PinballLoss_Normal(keras.losses.Loss):
    def __init__(self, quantiles, name="pinball_loss_normal"):
        super().__init__(name=name)
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        # y_pred is assumed to be the output of the DistributionLambda layer
        loc = y_pred.loc  # Mean of the normal distribution
        scale = y_pred.scale  # Scale of the normal distribution

        # Create a Normal distribution using TensorFlow Probability
        distribution = tfd.Normal(loc=loc, scale=scale)

        # Sample from the distribution
        pred_samples = distribution.sample(10000)

        # Compute the quantiles from the samples
        quantiles = tfp.stats.percentile(pred_samples, q=[q * 100 for q in self.quantiles], axis=0)

        y_true = tf.expand_dims(y_true, axis=-1)
        # Calculate pinball loss for each quantile
        losses = []
        for i, q in enumerate(self.quantiles):
            # Ensure y_true and quantiles[i] have compatible shapes

            error = y_true - quantiles[i]
            loss_q = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
            losses.append(loss_q)

        # Average the losses across all quantiles
        total_loss = tf.reduce_mean(losses)
        return total_loss

    def get_config(self):
        return {
            "num_quantiles": self.quantiles,
            "name": self.name,
        }

class WinklerLoss(keras.losses.Loss):
    def __init__(self, quantiles: List, name="winkler_loss"):
        super().__init__(name=name)
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        score = []
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles[:n // 2]):
            delta = tf.subtract(y_pred[:, :, -i - 1], y_pred[:, :, i])
            delta_u = (1 / q) * tf.subtract(y_true, y_pred[:, :, -i - 1]) * tf.cast(y_true > y_pred[:, :, -i - 1],
                                                                                    tf.float32)
            delta_l = (1 / q) * tf.subtract(y_pred[:, :, i], y_true) * tf.cast(y_true < y_pred[:, :, i], tf.float32)

            score.append(delta + delta_u + delta_l)
        score.append(tf.math.abs(tf.subtract(y_pred[:, :, n // 2], y_true)))
        L = tf.convert_to_tensor(score)
        total_loss = tf.reduce_mean(L)
        return total_loss
    def get_config(self):
        return {
            "num_quantiles": self.quantiles,
            "name": self.name,
        }

class CRPSLoss(keras.losses.Loss):
    def __init__(self, quantiles: List, name="custom_loss"):
        super().__init__(name=name)
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        # Compute the pinball error
        y_true_expanded = tf.expand_dims(y_true, axis=-1)
        pin_err = tf.abs(y_true_expanded - y_pred)

        # Compute the Winkler score component
        quantile_diff = tf.abs(tf.expand_dims(y_pred, axis=-1) - tf.expand_dims(y_pred, axis=-2))
        score = tf.reduce_mean(quantile_diff, axis=-1) / 2

        # Reduce mean over quantiles
        pin_err_mean = tf.reduce_mean(pin_err, axis=-1)
        score_mean = tf.reduce_mean(score, axis=-1)

        # Combine the error and score
        total_loss = pin_err_mean - score_mean

        return tf.reduce_mean(total_loss)

    def get_config(self):
        return {
            "num_quantiles": self.quantiles,
            "name": self.name,
        }

class Pinball_Delta_Loss(keras.losses.Loss):
    def __init__(self, quantiles: List, name="pinball_loss"):
        super().__init__(name=name)
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        loss = []
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles[:n//2]):
            error1 = tf.subtract(y_true, y_pred[:, :, i])
            error2 = tf.subtract(y_true, y_pred[:, :, -i-1])
            loss_q1 = tf.reduce_mean(tf.maximum(q * error1, (q - 1) * error1))
            q2 = self.quantiles[-i - 1]
            loss_q2 = tf.reduce_mean(tf.maximum(q2 * error2, (q2 - 1) * error2))
            hit_perc = tf.reduce_mean(tf.multiply(tf.cast(tf.greater(y_true,y_pred[:, :, i]),dtype=tf.float32),tf.cast(tf.greater(y_pred[:, :, -i-1], y_true),dtype=tf.float32)))
            loss_q = tf.multiply(tf.add(loss_q1, loss_q2), tf.abs(hit_perc-q2+q))/(q2-q)
            loss.append(loss_q)

        L = tf.convert_to_tensor(loss)
        total_loss = tf.reduce_mean(L)
        return total_loss