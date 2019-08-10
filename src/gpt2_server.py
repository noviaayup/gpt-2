#Function so that one session can be called multiple times. 
#Useful while multiple calls need to be done for embedding. 
import tensorflow as tf
import numpy as np
import os

import fire
import json
import traceback


import model, sample, encoder 

tf.logging.set_verbosity(tf.logging.ERROR)

model_name = '345M_285' # name of model folder under gpt2/models
enc = encoder.get_encoder(model_name)
hparams = model.default_hparams()
with open(os.path.join('models', model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

length=200

# lots of hard-coded params
seed=None
nsamples=1
batch_size=1
temperature=1
top_k=0
top_p=0.0

hist_len = 3 # feed the last n messages back into the model for context (its memory)

def gpt2_model():
    with tf.Graph().as_default():
        # input placeholder for [batch, input_tokens]
        context = tf.placeholder(tf.int32, [1, None]) # first dim is batch size (1), second is output text len
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=1,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        sess = tf.train.MonitoredSession()
        saver.restore(sess, ckpt)
        return lambda x: sess.run(output, feed_dict={ context: x})[:, len(x[0]):] # len(x[0]) returns everything after input prompt

model_fn = gpt2_model()


import socket
import sys

HOST = 'localhost'	# Symbolic name, meaning all available interfaces
PORT = 9999	# Arbitrary non-privileged port

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f'Socket created on {HOST}:{PORT}')

#Bind socket to local host and port
try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
    sys.exit()
    
print('Socket bind complete')

#Start listening on socket
s.listen(10)
print('Socket now listening')

#now keep talking with the client
try:
    while True:
        #wait to accept a connection - blocking call
        conn, addr = s.accept()
        print('Connected with ' + addr[0] + ':' + str(addr[1]))
        hist_buf = []
        while True:
            input_msg = conn.recv(4096).decode("utf-8").strip()
            if input_msg == 'cc':
                conn.close()
                print('Connection closed with ' + addr[0] + ':' + str(addr[1]))
                break
            print('>>Recvd msg:'+input_msg)
            hist = ''.join(hist_buf[-hist_len:]) # run the last few messages back into the model for context
            context_tokens = enc.encode(hist+"[INPUT]: "+input_msg+"\n\n[Joe Rogan]:")
            print('running through gpt2...')
            out_tokens = model_fn([context_tokens]) # passes this to the lambda
            out_str = enc.decode(out_tokens[0])
            trunc_text = out_str.split('\n')[0]
            # print(out_str)
            print(trunc_text)
            return_string = "\nJoe Rogan:"+ trunc_text +'\n\n'

            conn.send(return_string.encode('utf-8')) # send result to client

            hist_buf.append("[INPUT]: "+input_msg+"\n\n[Joe Rogan]:"+trunc_text+"\n\n") # store history of convo
except Exception as e:
    print(traceback.format_exc())
    s.close()
    
s.close()