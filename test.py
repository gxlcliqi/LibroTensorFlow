import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


num_vectors = 10
num_clusters = 3
num_steps = 100
vector_values = []
for i in range(num_vectors):
        vector_values.append([i, i])

vectors = tf.constant(vector_values)
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),
                                 [0,0],[num_clusters,-1]))
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

print(expanded_vectors.get_shape())
print(expanded_centroids.get_shape())

distances = tf.reduce_sum(
  tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)

assignments = tf.argmin(distances, 0)
b = tf.equal(assignments, 2)
c = tf.reshape(
    tf.where(
        tf.equal(assignments, 2)
    ), [1, -1])
d = tf.gather(vectors,
          tf.reshape(
              tf.where(
                  tf.equal(assignments, 2)
              ), [1, -1])
          )
e = [tf.reduce_mean(
    tf.gather(vectors,
              tf.reshape(
                  tf.where(
                      tf.equal(assignments, tt)
                  ), [1, -1])
              ), reduction_indices=[1]) for tt in range(num_clusters)]

f = tf.concat([
  tf.reduce_mean(
      tf.gather(vectors,
                tf.reshape(
                  tf.where(
                    tf.equal(assignments, kk)
                  ), [1, -1])), reduction_indices=[1]) for kk in range(num_clusters)], 0)

init_op = tf.global_variables_initializer()

#with tf.Session('local') as sess:
sess = tf.Session()
sess.run(init_op)
print("test")
print(sess.run(expanded_vectors))
print("test2")
print(sess.run(expanded_centroids))

print(sess.run(distances))
print(sess.run(assignments))
print(sess.run(b))
print(sess.run(c))
print(sess.run(d))
print(sess.run(e))
print(sess.run(f))

print(sess.run(centroids))