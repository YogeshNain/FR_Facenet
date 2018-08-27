#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 21:25:53 2018

@author: yogi
"""

import cv2 
import FaceRecognition as FR
import argparse
import sys

def drawonImage(img,faces):
    if faces is not None:
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,160,20))
            if face.name is not None:
                if(face.confidence > 0.50):
                    cv2.putText(img,face.name,(bbox[0],bbox[3]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
                cv2.putText(img,str(face.confidence),(bbox[0],bbox[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))


def runEngine(argus):
    
    print('Initalizing FR Engine')
    FREngine = FR.Recognition()
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
    scale = argus.scale
    margin = argus.margin
    
    while True:
        _,img = videoStream.read()
        img = cv2.resize(img,(0,0),fx=scale,fy=scale)
        faces = FREngine.identify(img,scale)
        drawonImage(img,faces)
        
        cv2.imshow('Video',img)
        
        wrimg = cv2.resize(img,(960,540))
        
        videoOut.write(wrimg)
        key= cv2.waitKey(10)
        
        if key == ord('q'):
            break
        if key == ord('p'):
            key = cv2.waitKey()
        
    cv2.destroyAllWindows()

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
    
    runEngine(progArguments)