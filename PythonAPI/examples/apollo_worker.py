# encoding: utf-8
#
#   Custom routing Router to Dealer
#
#   Author: Jeremy Avnet (brainsik) <spork(dash)zmq(at)theory(dot)org>
#

import sys
import time
import random
from threading import Thread

import zmq


# We have two workers, here we copy the code, normally these would
# run on different boxes...
#
def worker_a(context=None):
    context = context or zmq.Context.instance()
    
    # Socket to receive messages on
    receiver = context.socket(zmq.PULL)
    receiver.connect("tcp://localhost:5556")

    # Socket to send messages to
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.CONFLATE, 1)  # last msg only.
    socket.connect("tcp://localhost:5557")

    # Process tasks forever
    while True:
    
        s = receiver.recv_string()

        # Simple progress indicator for the viewer
        sys.stdout.write('.')
        sys.stdout.flush()

        #Do the work
        #time.sleep(int(s)*0.001)
        
        # Send results to sink
        socket.send_string(s)



def worker_b(context=None):
    context = context or zmq.Context.instance()
    
    # Socket to receive messages on
    receiver = context.socket(zmq.PULL)
    receiver.connect("tcp://localhost:5558")

    # Socket to send messages to
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.CONFLATE, 1)  # last msg only.
    socket.connect("tcp://localhost:5559")

    # Process tasks forever
    while True:
        
        s = receiver.recv_string()
        
        # Simple progress indicator for the viewer
        sys.stdout.write('*')
        sys.stdout.flush()

        #Do the work
        #time.sleep(int(s)*0.001)
        
        # Send results to sink
        socket.send_string(s)
        

#context = zmq.Context.instance()
#client = context.socket(zmq.ROUTER)
#client.bind("ipc://routing.ipc")

Thread(target=worker_a).start()
Thread(target=worker_b).start()

## Wait for threads to stabilize
#time.sleep(1)

## Send 10 tasks scattered to A twice as often as B
#for _ in range(10):
    ## Send two message parts, first the address...
    #ident = random.choice([b'A', b'A', b'B'])
    ## And then the workload
    #work = b"This is the workload"
    #client.send_multipart([ident, work])

#client.send_multipart([b'A', b'END'])
#client.send_multipart([b'B', b'END'])
