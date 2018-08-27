#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 17:03:36 2018

@author: yogi
"""

import cv2
import sys
import argparse
import numpy as np
import tensorflow as tf
from detect_face import detect_face
from facenetpkg import facenet


#Configuration data variables.
detectModel = 'cfg/model/'
faceNetModel = 'cfg/model/M1.pb'
faceClassifier = 'cfg/model/classifier.pkl'
#alignModel = 'cfg/model/faceLandMark.dat'

facenet.load_model(faceNetModel)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]


faceMinSize = 40
faceThreshold = [0.9, 0.7, 0.8] # 3 step threshold for face detection
scaleFactor = 0.6 


def runFREngine(argus):
    print('\t\t\t |||| Face Recognition Engine ||||')
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, detectModel)
    inputStream = argus.inputFile
    outputStream = argus.outFile
    if outputStream == None:
        outputStream = "_test"
    
    if inputStream == '0':
        inputStream = 0
    print('Processing : %s'%inputStream)
    videoStream = cv2.VideoCapture(inputStream)    
    if not videoStream.isOpened():
        print('can not open video stream : %s'%inputStream)
        return 
    
    forcc = cv2.VideoWriter_fourcc(*'XVID')
    videoOut = cv2.VideoWriter(outputStream+".avi",forcc,15,(960,540))
    
    totalImages= 0
    scale = argus.scale
    margin = argus.margin
    alignFaceSize = 160
    leftEyeDist = (0.35,0.35)
    while(True):
        _,frame = videoStream.read()
        drimg = frame
        startTime = cv2.getTickCount()
        frame = cv2.resize(frame,(0,0),fx=scale,fy=scale)
        bbox, lpoint = detect_face.detect_face(frame,argus.minSizeFace,pnet,rnet,onet,faceThreshold,scaleFactor)
        endTimeDetection = cv2.getTickCount()
        timedetection = (endTimeDetection - startTime)/cv2.getTickFrequency()
        cv2.putText(drimg,str(format(timedetection*1000,'.2f')),(20,50),cv2.FONT_HERSHEY_PLAIN,2,(200,0,20))
        cv2.putText(drimg,str(format(1/timedetection,'.2f')),(20,70),cv2.FONT_HERSHEY_PLAIN,2,(200,200,0))
        nr_of_Faces = bbox.shape[0]
        #print(lpoint.shape[0])
        cv2.putText(drimg,str(nr_of_Faces),(20,20),cv2.FONT_HERSHEY_PLAIN,2,(200,0,200))
        iml = 0
        imgSize = np.asarray(drimg.shape)[0:2]
        bbox = deflate_bbox(bbox,scale,margin,imgSize)
        for br in bbox:
            x1,x2,y1,y2 = pad_img_to_fit_bbox(drimg,int(br[0]),int(br[2]),int(br[1]),int(br[3]))
            faceCrop = drimg[y1:y2,x1:x2]
            #cv2.rectangle(drimg,(int(br[0]),int(br[1])),(int(br[2]),int(br[3])),(0,0,255))
            cv2.imshow(str(iml),faceCrop)
            wrFile = "Faces/"+str(startTime+iml)+".png"
            cv2.imwrite(wrFile,faceCrop)
            iml+=1
            angle= 0.0
            '''for pt in lpoint.T:
                lt = (int(pt[0]/scale),int(pt[5]/scale))
                rt = (int(pt[1]/scale),int(pt[6]/scale))
                dy = rt[1] - lt[1]
                dx = rt[0] - lt[0]
                angle = np.degrees(np.arctan2(dy,dx))
                print('Angle : ',angle)
            if(angle != 0.0):
                eyeC = ((lt[0] + rt[0])//2,(lt[1]+rt[1])//2)
                reye = 1.0 - leftEyeDist[0]
                dist = np.sqrt((dx**2)+(dy**2))
                desdist = (reye - leftEyeDist[0])*alignFaceSize
                fSc = desdist/dist
                #print(fSc)
                M = cv2.getRotationMatrix2D( eyeC,float(angle),fSc)
                rows,cols,_ = faceCrop.shape
                align = cv2.warpAffine(faceCrop,M,(cols,rows))
                cv2.imshow("Faces_AL",align)'''
            #print(int(br[0]))
        cv2.imshow('video',drimg)
        #cv2.waitKey()
       
        
        resimg = cv2.resize(drimg,(940,540))
        videoOut.write(resimg)
        totalImages +=1
        key = cv2.waitKey(20)
        
        if key == ord('q'):
            break
        if key == ord('p'):
            key = cv2.waitKey()

    cv2.destroyAllWindows()            

def get_aligned_image(img,landmark,faceSize,scale):
    
    '''for pt in landmark.T:
        for c in range(5):
            cv2.circle(img,(int(pt[c]/scale),int(pt[c+5]/scale)),1,(255,0,0),2)
    '''
    #
        
def deflate_bbox(bbox,scale,margin,imgSize):
    faceBoxes = []
    for br in bbox:
        br[0] = br[0]/scale
        br[0] = np.maximum(br[0]-margin/2,0)
        br[1] = br[1]/scale
        br[1] = np.maximum(br[1]-margin/2,0)
        br[2] = br[2]/scale
        br[2] = np.minimum(br[2]+margin/2,imgSize[1])
        br[3] = br[3]/scale
        br[3] = np.minimum(br[3]+margin/2,imgSize[0])
        faceBoxes.append(br)
    return faceBoxes
        



        
def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
        img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
                   (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")
        y1 += np.abs(np.minimum(0, y1))
        y2 += np.abs(np.minimum(0, y1))
        x1 += np.abs(np.minimum(0, x1))
        x2 += np.abs(np.minimum(0, x1))
        return x1, x2, y1, y2

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--inputFile',type=str,help='video or image file path')
    parser.add_argument('-o','--outFile',type=str,help='output file name <.avi>/<.png>')
    parser.add_argument('-w','--weights',type=str,help='Weight folder having weight file for face detection and recognition')
    parser.add_argument('-s','--minSizeFace',type=int,help='min size of face to detect')
    parser.add_argument('-m','--margin',type=int,help='margin to cut arround face area',default=20)
    parser.add_argument('-t','--threshold',type=float,help='threshold for recognition')
    parser.add_argument('--scale',type=float,help='scale of image for finding face',default=0.5)
    progArguments = parser.parse_args(sys.argv[1:])
    
    runFREngine(progArguments)
                     