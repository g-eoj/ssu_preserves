import os
import pylab
import sys
import tensorflow as tf
from os import listdir

# location of the labels used in the retrained model
labels = "./tmp/output_labels.txt"
# location of the graph for the retrained model
model = "./tmp/output_graph.pb"

def classify(image_path, graph_def, label_lines):
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        result = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            result.append((human_string, score))
            #print('%s (score = %.5f)' % (human_string, score))

    return result 

# change this as you see fit
folder_path = sys.argv[1]

# Loads label file for the retrained model, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile(labels)]

# Unpersists graph from file
with tf.gfile.FastGFile(model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

print("getting filename list...")
filenames = os.listdir(folder_path)
print("classifying turkey jpgs...")
correct = 0
index = 0
while index < len(filenames):
        filename = filenames[index];
        index += 1
        if filename.endswith(".JPG"):
            image_path = folder_path + filename
            result = classify(image_path, graph_def, label_lines)
            # top 1
            if result[0][0] == 'turkey':
                correct += 1
            print('%s (score = %.5f) | filename: %s' % (result[0][0], result[0][1], filename))


print("Accuracy:", float(correct) / float(index))
