#/usr/bin/env python

import time
import tensorflow as tf

results = []

for tau in range(4, 13):
    k = 2**tau
    a = tf.random.uniform(shape=[k,k], minval=1, maxval=2,dtype=tf.float16)
    b = tf.random.uniform(shape=[k,k], minval=1, maxval=2,dtype=tf.float16)

    cpu_slot = 0
    gpu_slot = 0

    cpu_time = []
    gpu_time = []

    # Using CPU at slot 0
    print("Running CPU matmul with matrixsize: ", k)
    with tf.device('/CPU:' + str(cpu_slot)):
        for x in range(0, 9):
            start = time.time()
            c1 = tf.matmul(a,b)
            #print("Time on CPU:")
            end = time.time() - start
            cpu_time.append(end)
            #print(end)

    # Using the GPU at slot 0
    print("Running GPU matmul with matrixsize: ", k)
    with tf.device('/GPU:' + str(gpu_slot)):
        for x in range(0, 9):
            start = time.time()
            c2 = tf.matmul(a,b)
            #print("Time on GPU:")
            end = time.time() - start
            gpu_time.append(end)
            #print(end)

    cpu_avg = sum(cpu_time)/len(cpu_time)
    gpu_avg = sum(gpu_time)/len(gpu_time)
    speedup = cpu_avg / gpu_avg;

    localResult = {
        'matrixsize': k,
        'cpu' : cpu_avg,
        'gpu' : gpu_avg,
        'speedup' : speedup
    }

    results.append(localResult)

for res in results:
    print(res)

