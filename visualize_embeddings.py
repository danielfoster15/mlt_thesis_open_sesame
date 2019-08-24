import os
import argparse

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import TKvector


def write_meta_file(embeddings, meta_file):
    with open(meta_file, 'w') as f:
        for word in embeddings.vocab:
            f.write(word + "\n")


def visualize_embeddings(embedding_file, log_dir):
    embeddings = TKvector.TKWV(embedding_file)
    print(embeddings.vectors)

    # write file containing labels for vectors
    os.makedirs(log_dir)
    meta_filename = "meta_dontuse.tsv"
    write_meta_file(embeddings, os.path.join(log_dir, meta_filename))

    # write model checkpoint containing the embeddings
    images = tf.Variable(embeddings.vectors, name='embeddings')
    with tf.Session() as sess:
        saver = tf.train.Saver([images])
        sess.run(images.initializer)
        saver.save(sess, os.path.join(log_dir, 'embeddings.ckpt'))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    embedding.metadata_path = meta_filename
    projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Creates model files for tensorboard embeddings projector.")
    parser.add_argument("embedding_file", help="Path to embeddings file")
    parser.add_argument("log_dir", help="Destination directory")
    args = parser.parse_args()

    visualize_embeddings(args.embedding_file, args.log_dir)
