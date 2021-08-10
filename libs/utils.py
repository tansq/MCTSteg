import glob
import random
import math
import hashlib
import logging
import argparse
import subprocess
import numpy as np
import scipy
import matlab
import matlab.engine
import tensorflow as tf
import os
import cv2
import sys
import argparse
from SRNet import SRNet
from scipy import misc, io

PI = math.pi
sys.setrecursionlimit(20000)
"seed is img_id"
eng = matlab.engine.start_matlab()
g_xunet = tf.Graph()
g_srnet = tf.Graph()
g_dct = tf.Graph()
QF = '75'
dataQ = io.loadmat('libs/quant_tables/quant_' + QF + '.mat')
quant = dataQ['quant']

sorted_idx = None
state = None

def set_idx(idx):
    global sorted_idx
    sorted_idx = idx


def get_idx():
    global sorted_idx
    return sorted_idx


def set_state(sublattice_size):
    global state
    state = np.random.randint(-1, 2, [sublattice_size, sublattice_size])
    set_zero()


def set_state_element(x,y,e):
    global state
    state[x][y] = e


def get_state():
    global state
    return state


def set_zero():
    global state
    global sorted_idx
    for i in np.arange(-1,-128*96,-1):
        x = int(sorted_idx[i]/128)
        y = int(sorted_idx[i]%128)
        state[x][y] = 0
    return state

def get_portion_idx():
    x_1 = []
    x_2 = []
    for outside_idx in range(0,256,8):
        for inside_idx in range(outside_idx, outside_idx+8):
            if (outside_idx/8)%2 == 0:
                x_1.append(inside_idx)
            else:
                x_2.append(inside_idx)
    idx_list = np.zeros([2,128])
    idx_list[0,:] = x_1
    idx_list[1,:] = x_2
    idx_list = idx_list.astype(np.int)
    return idx_list


def get_idx_portion(cover, idx, idx_list):
    port = cover[np.ix_(idx_list[idx/2], idx_list[idx%2])] 
    return port


def get_portion(mode,cover):
    p1 = cover[mode == 1]
    p2 = cover[mode == 2]
    p3 = cover[mode == 3]
    p4 = cover[mode == 4]
    p1 = p1.reshape([128 , 128])
    p2 = p2.reshape([128 , 128])
    p3 = p3.reshape([128 , 128])
    p4 = p4.reshape([128 , 128])
    return [p1,p2,p3,p4]


def get_portion_mode():
    p1 = np.ones([8,8])
    p2 = p1+1
    p3 = p1+2
    p4 = p1+3
    p12 = np.hstack((p1,p2))
    p34 = np.hstack((p3, p4))
    p = np.vstack([p12,p34])
    p = np.hstack([p,p,p,p,p,p,p,p])
    p = np.vstack([p,p,p,p,p,p,p,p])
    p = np.hstack([p, p])
    p = np.vstack([p, p])
    return p


def DCT_layer(x):
    table = tf.constant(quant.astype(np.float32))
    table1 = tf.expand_dims(table, 0)
    table2 = tf.expand_dims(table1, 3)
    tables = tf.tile(table2, [1, 1, 1, 1])

    xT = tf.multiply(x, tables)
    IDCTBase = np.zeros([8, 8, 1, 64], dtype=np.float32)  # [height,width,input,output]
    w = np.ones([8], dtype=np.float32)
    w[0] = 1.0 / math.sqrt(2.0)
    for i in range(0, 8):
        for j in range(0, 8):
            for k in range(0, 8):
                for l in range(0, 8):
                    IDCTBase[k, l, :, i * 8 + j] = w[k] * w[l] / 4.0 * math.cos(PI / 16.0 * k * (2 * i + 1)) * math.cos(
                        PI / 16.0 * l * (2 * j + 1))

    IDCTKernel = tf.Variable(IDCTBase, name="IDCTKenel", trainable=False)
    # Pixel = tf.nn.relu(tf.nn.conv2d(xT,IDCTKernel,[1,8,8,1],'VALID',name="Pixel")+128)
    Pixel = tf.nn.conv2d(xT, IDCTKernel, [1, 8, 8, 1], 'VALID', name="Pixel") + 128
    Input = tf.depth_to_space(Pixel, 8)
    return Input
    # DCTBase = np.zeros([4, 4, 1, 16], dtype=np.float32)  # [height,width,input,output]
    # u = np.ones([4], dtype=np.float32) * math.sqrt(2.0 / 4.0)
    # u[0] = math.sqrt(1.0 / 4.0)
    # for i in range(0, 4):
    #     for j in range(0, 4):
    #         for k in range(0, 4):
    #             for l in range(0, 4):
    #                 DCTBase[i, j, :, k * 4 + l] = u[k] * u[l] * math.cos(PI / 8.0 * k * (2 * i + 1)) * math.cos(
    #                     PI / 8.0 * l * (2 * j + 1))
    #
    # DCTKernel = tf.Variable(DCTBase, name="DCTKenel", trainable=False)
    # DCT = tf.abs(tf.nn.conv2d(Input, DCTKernel, [1, 1, 1, 1], 'VALID', name="DCT"))
    # DCT_Trunc = -(tf.nn.relu(-DCT + 8) - 8)  # Trancation operation
    # return DCT_Trunc


def buildDCT():
    with g_dct.as_default():
        DCT = tf.placeholder(tf.float32, shape=[1, 256, 256, 1])
        res = DCT_layer(DCT)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        return sess, res, DCT


def dct2spacial(coeffs, sess, res, DCT):
    coe = np.ones([1, 256, 256, 1])
    coe[0, :, :, 0] = coeffs
    res = sess.run(res, feed_dict={DCT: coe})
    return res[0,:,:,0]


def load_mat(path):
    m = io.loadmat(path)['im']
    return m


def save_mat(mat, path):
    io.savemat(path, {'im': mat})


def read_jpeg(path):
    S_STRUCT, S_SPATIAL, S_COEFFS, S_QUANT = eng.read_jpeg(path, nargout=4)
    return S_STRUCT, np.array(S_SPATIAL), np.array(S_COEFFS), np.array(S_QUANT)


def write_jpeg(S_COEFFS, path):
    S_COEFFS = S_COEFFS.tolist()
    path = eng.write_jpeg(matlab.double(S_COEFFS), path)


def JUNIWARD(cover_path, payload):
    print("Computing JUNIWARD cost.")
    C_STRUCT, C_SPATIAL, C_COEFFS, C_QUANT = eng.read_jpeg(cover_path, nargout=4)
    cost, nzAC = eng.f_cal_cost_JUNIWARD(matlab.double(C_SPATIAL), matlab.double(C_COEFFS), matlab.double(C_QUANT),
                                         payload, nargout=2)
    return np.array(cost), nzAC


def SUNIWARD(cover, payload):
    print("Computing SUNIWARD cost.")
    cover = cover.tolist()
    cost = eng.f_cal_cost_SUNIWARD(matlab.double(cover), payload)
    return np.array(cost)


def HILL(cover):
    print("Computing HILL cost.")
    cover = cover.tolist()
    cost = eng.f_cal_cost_HILL(matlab.double(cover))
    return np.array(cost)


def embed(cover, change, cost_map, seed, w, payload):
    
    stego = simu(cover, change, cost_map, seed, w, payload)
    return stego


def STC(cover, change, cost_map, w):
    print("Generating Stego with STC")
    cover = cover.tolist()
    cost_map = cost_map.tolist()
    change = change.tolist()
    stego = eng.f_stc_emb_filter(matlab.double(cover), matlab.double(change), matlab.double(cost_map), w)
    stego = np.array(stego)
    return stego


def simu(cover, change, cost_map, seed, w, payload):
    print("Generating Stego with simulator")

    cover = cover.tolist()
    cost_map = cost_map.tolist()
    change = change.tolist()
    stego = eng.f_emb_filter(matlab.double(cover), matlab.double(change), matlab.double(cost_map), seed, w, payload)
    stego = np.array(stego)
    return stego


def embed_j(coef, change, cost_map, alpha, payload):
    [l, w] = coef.shape
    rho_P1 = np.zeros([l, w])
    rho_M1 = np.zeros([l, w])
    rho_P1[:,:] = cost_map
    rho_M1[:,:] = cost_map
    rho_P1[change == 1] = rho_P1[change == 1] / alpha
    rho_M1[change == -1] = rho_M1[change == -1] / alpha
    coef = coef.tolist()
    rho_P1 = rho_P1.tolist()
    rho_M1 = rho_M1.tolist()
    nzAC = eng.f_cal_nzAC(matlab.double(coef))
    embedded_coef = eng.f_JPEG_EmbeddingSimulator(matlab.double(coef), matlab.double(rho_P1), matlab.double(rho_M1),
                                                  nzAC, payload, nargout=1)
    embedded_coef = np.array(embedded_coef)
    embedded_coef = embedded_coef.astype(np.int)
    return embedded_coef


def CMD(cover):
    cover = cover.tolist()
    # row = [2]
    # row = row.to_list()
    # lin = [2]
    # lin = lin.to_list()
    # payload = [0.4]
    # payload = payload.to_list()
    stego = eng.f_related_embed(matlab.double(cover), seed)
    return np.array(stego)


def build_xunet(model_path):
    with g_xunet.as_default():
        sess = tf.Session()
        from xunet_model import xunet
        model = xunet(sess, model_path)
        return model


def build_srnet(model_path):
    with g_srnet.as_default():
        model = SRNet(False, 'NCHW')
        batch = tf.placeholder(tf.uint8, shape=[2, 256, 256, 1])
        label = tf.placeholder(tf.int64, shape=[2, ])
        model._build_model(batch)
        loss, accuracy = model._build_losses(label)
        con = tf.nn.softmax(model.outputs)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10000)
        sess = tf.Session()
        sess.run(init_op)
        saver.restore(sess, model_path)
        return model, con, sess


def classify(model, model_type, model_path, cover, stego):
    if model_type == "xunet":
        with g_xunet.as_default():
            c = np.empty((1, 256, 256, 1), dtype='uint8')
            s = np.empty((1, 256, 256, 1), dtype='uint8')
            c[0, :, :, 0] = cover
            s[0, :, :, 0] = stego
            confidence = model.get_confidence(c, s)
            print(confidence)
            return confidence[1]
    else:
        batch = np.empty((2, 256, 256, 1), dtype='uint8')
        batch[0, :, :, 0] = cover
        batch[1, :, :, 0] = stego
        label_batch = np.array([0, 1], dtype='uint8')
        with g_srnet.as_default():
            con = tf.nn.softmax(model.outputs)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            saver = tf.train.Saver(max_to_keep=10000)
            with tf.Session() as sess:
                sess.run(init_op)
                saver.restore(sess, model_path)
                res = sess.run(con, feed_dict={model.inputs: batch, model.labels: label_batch})
                print(res)
                return res[1][0]


def get_confidence(model, con, sess, cover, stego):
    batch = np.empty((2, 256, 256, 1), dtype='uint8')
    batch[0, :, :, 0] = cover
    batch[1, :, :, 0] = stego
    label_batch = np.array([0, 1], dtype='uint8')
    res = sess.run(con, feed_dict={model.inputs: batch, model.labels: label_batch})
    print(res)
    return res[1][0]


def normalization(data):
    data = np.array(data)
    data /= data.sum()
    return data


def save(name, data, label):
    print("Saved data!")

    l = len(data)
    for i in range(l):
        x = data[i]
        y = label[i]
        print("save " + str(i) + " label " + str(y))
        path = "./train/" + name + "_" + str(i) + '.npy'
        np.save(path, x)
        with open("label.txt", 'a+') as f:
            f.writelines(path + " " + str(y) + '\n')
    '''

    path = "./train_3/" + name + "_" + str(idx) + '.npy'
    np.save(path, data)
    with open("label_3.txt", 'a+') as f:
        f.writelines(path + " " + str(label) + '\n')
    '''


def get_neighbor(i, j, size):
    nei = pad_modi[i - size:i + size + 1, j - size:j + size + 1]
    return nei


def get_cost_neighbor(i, j, size):
    nei_cost = pad_cost[i - size:i + size + 1, j - size:j + size + 1]
    return nei_cost



def dist(mtrx1, mtrx2):
    return np.sqrt(np.sum(np.square(mtrx1 - mtrx2)))
