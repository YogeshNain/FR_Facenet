#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 20:07:31 2018

@author: yogi
"""

from detect_face import detect_face
from scipy import misc
from facenetpkg import facenet
import pickle
import numpy as np
import tensorflow as tf

gpu_memory_fraction = 0.3
classifier_model = 'cfg/model/class.pkl'
facenetModel = 'cfg/model/M1.pb'
mtcnnModel = 'cfg/model/'
detect_Scale = 0.5

class Face:
    def __init__(self):
        self.name = None
        self.bbox = None
        self.image = None
        self.embeding = None
        self.cofidence = None
        
class Recognition:
    def __init__(self):
        self.detector = Detection()
        self.encoding = Encoder()
        self.identifier = Identify()
    def add_identity(self,image,profileName):
        faces = self.detector.find_faces(image)
        if len(faces) == 1:
            face = faces[0]
            face.name = profileName
            face.embedding = self.encoding.get_embedding(face)
            return faces
    def identify(self,image,scale):
        detect_Scale = scale
        faces = self.detector.find_faces(image)
        for i, face in enumerate(faces):
            face.embedding = self.encoding.get_embedding(face)
            #face.name,face.confidence = self.identifier.identify(face)
        return faces
    def readIdenties(self,idsPath):
        dataset = facenet.get_dataset(idsPath)
        print('Found ids :%d'%len(dataset))
        return dataset
        

class Identify():
    def __init__(self):
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)
         
    def identify(self,face):
        if face.embedding is not None:
            predictions = self.model.predict_proba([face.embedding])
            #print(predictions)
            best_class = np.argmax(predictions,axis=1)
            confidence = predictions.T[best_class[0]]
            #print("best : %f : %d "%(predictions.T[best_class[0]],best_class[0]))
            return self.class_names[best_class[0]], confidence
        
class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenetModel)
    def get_embedding(self,face):
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        prewhiten_face = facenet.prewhiten(face.image)
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]
        
            
        
class Detection:
    # face detection parameters
    minsize = 60  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return detect_face.create_mtcnn(sess, mtcnnModel)

    def find_faces(self, image):
        faces = []
        bounding_boxes, _ = detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.bbox = np.zeros(4, dtype=np.int32)
            
            img_size = np.asarray(image.shape)[0:2]
            face.bbox[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bbox[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bbox[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bbox[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bbox[1]:face.bbox[3], face.bbox[0]:face.bbox[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
            faces.append(face)
        return faces
            