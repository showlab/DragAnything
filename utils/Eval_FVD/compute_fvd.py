"""
Computes the Frechet Video Distance between videos in two directories.
The videos need to be in GIF format. The number of videos in each directory needs to be a multiple
of 16 (remainders will be discarded).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from tqdm import tqdm
import numpy as np

import tensorflow.compat.v1 as tf
import frechet_video_distance as fvd

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('-d0','--dir0', type=str, default='video_dir0')
argparser.add_argument('-d1','--dir1', type=str, default='video_dir0')


# Number of videos must be divisible by 16.
VIDEO_BATCH_SIZE = 16


def main(argv):

  args = argv[0]

  # read file lists from directories
  dir0_gif_paths = [f for f in os.listdir(args.dir0) if f.endswith('.gif')]
  dir0_gif_paths = [os.path.join(args.dir0, f) for f in dir0_gif_paths]
  dir1_gif_paths = [f for f in os.listdir(args.dir1) if f.endswith('.gif')]
  dir1_gif_paths = [os.path.join(args.dir1, f) for f in dir1_gif_paths]
  # assert number of videos to be divisible by 16
  remainder_dir0 = len(dir0_gif_paths) % VIDEO_BATCH_SIZE
  dir0_gif_paths = dir0_gif_paths[:-remainder_dir0]
  remainder_dir1 = len(dir1_gif_paths) % VIDEO_BATCH_SIZE
  dir1_gif_paths = dir1_gif_paths[:-remainder_dir1]

  # loop over video dirs in batches of 16, compute and assemble activations (id3_embedding)
  dir0_embeddings, dir1_embeddings = [], []
  # graph0_initialized, graph1_initialized = False, False
  dir0_embeddings_file = os.path.join(args.dir0, 'id3_embeddings.npy')
  dir1_embeddings_file = os.path.join(args.dir1, 'id3_embeddings.npy')

  # --- dir0 ID3 embeddings
  if os.path.exists(dir0_embeddings_file):
    with open(dir0_embeddings_file, 'rb') as fp:
      dir0_embeddings = np.load(fp)
    print(f">>> Found stored ID3 activations for videos in {args.dir0} in {dir0_embeddings_file}.")
  else:
    print(f">>> Computing ID3 activations for videos in {args.dir0}...")
    for batch_start_idx in tqdm(range(0, len(dir0_gif_paths), VIDEO_BATCH_SIZE)):
      with tf.Graph().as_default():
        # load batch of videos from GIFs and represent as tensor
        dir0_videos = tf.stack(
            [tf.io.decode_gif(tf.io.read_file(f)) \
            for f in dir0_gif_paths[batch_start_idx:batch_start_idx+VIDEO_BATCH_SIZE]])
        with tf.Session() as sess:
          dir0_tensor = sess.run(dir0_videos)
        # define placeholder for subsequent feeding
        ph_dir0_videos = tf.placeholder(shape=[*dir0_tensor.shape], dtype=tf.uint8)
        # calculate embeddings
        id3_embeddings = fvd.create_id3_embedding(fvd.preprocess(ph_dir0_videos, (224, 224)))
        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(tf.tables_initializer())
          dir0_embeddings.append(
              sess.run(id3_embeddings, feed_dict={ph_dir0_videos : dir0_tensor}))
    dir0_embeddings = np.concatenate(dir0_embeddings, axis=0)
    with open(dir0_embeddings_file, 'wb') as fp:
      np.save(fp, dir0_embeddings)
      print(f">>> Saved ID3 embeddings for lookup in {dir0_embeddings_file}")
  print(f">>> Embedding matrix: {dir0_embeddings.shape}")

  # --- dir1 ID3 embeddings
  if os.path.exists(dir1_embeddings_file):
    with open(dir1_embeddings_file, 'rb') as fp:
      dir1_embeddings = np.load(fp)
    print(f">>> Found stored ID3 activations for videos in {args.dir1} in {dir1_embeddings_file}.")
  else:
    print(f">>> Computing ID3 activations for videos in {args.dir1}...")
    for batch_start_idx in tqdm(range(0, len(dir1_gif_paths), VIDEO_BATCH_SIZE)):
      with tf.Graph().as_default():
        # load batch of videos from GIFs and represent as tensor
        dir1_videos = tf.stack(
            [tf.io.decode_gif(tf.io.read_file(f)) \
            for f in dir1_gif_paths[batch_start_idx:batch_start_idx+VIDEO_BATCH_SIZE]])
        with tf.Session() as sess:
          dir1_tensor = sess.run(dir1_videos)
        # define placeholder for subsequent feeding
        ph_dir1_videos = tf.placeholder(shape=[*dir1_tensor.shape], dtype=tf.uint8)
        # calculate embeddings
        id3_embeddings = fvd.create_id3_embedding(fvd.preprocess(ph_dir1_videos, (224, 224)))
        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(tf.tables_initializer())
          dir1_embeddings.append(
              sess.run(id3_embeddings, feed_dict={ph_dir1_videos : dir1_tensor}))
    dir1_embeddings = np.concatenate(dir1_embeddings, axis=0)
    with open(dir1_embeddings_file, 'wb') as fp:
      np.save(fp, dir1_embeddings)
      print(f">>> Saved ID3 embeddings for lookup in {dir1_embeddings_file}")
  print(f">>> Embedding matrix: {dir1_embeddings.shape}")
  
  # --- final FVD
  with tf.Graph().as_default():
    print(">>> Computing FVD...")
    result = fvd.calculate_fvd(dir0_embeddings, dir1_embeddings)
    with tf.Session() as sess:
      print(">>> FVD is: %.2f." % sess.run(result))


if __name__ == "__main__":
  args = argparser.parse_args()
  argv = [args]
  tf.app.run(main=main, argv=argv)