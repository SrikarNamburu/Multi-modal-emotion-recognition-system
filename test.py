# Importing dependencies
import moviepy
import moviepy.editor
import numpy as np
import cv2 
import os
import random
import librosa
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from keras.models import load_model
import pickle
import argparse

def _main():

    # parse command line arguments
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument(
        '--path_to_test_file', default = 'Data/testing/happy.mp4',type=str,
        help='The path to the input .mp4 on which emotion detection will be performed on.')
    parser.add_argument(
        '--path_to_save_audio_file', default='Data/testing/audio',type=str,
        help="The path where extracted audio file will be saved")
    parser.add_argument(
        '--path_to_save_frames', default='Data/testing/frames',type=str,
        help="The path where extracted frames will be saved")

    args = vars(parser.parse_args())

    Testfilepath = args['path_to_test_file']
    AudioFilepath = args['path_to_save_audio_file']
    Framespath = args['path_to_save_frames']


    # Extracting .wav and frames from .mp4 file

    def extractDetails(path):
        video = moviepy.editor.VideoFileClip(path)
        audio = video.audio
        audio.write_audiofile(AudioFilepath+'/audio.wav')
        vidObj = cv2.VideoCapture(path) 
        count = 0
        # checks whether frames were extracted 
        success = 1
        while success: 
            try:             
                # vidObj object calls read and function extract frames 
                success, image = vidObj.read() 
                # Saves the frames with frame-count 
                cv2.imwrite(Framespath+"//frame%d.jpg" % count, image) 
                count += 1
            except:
                Done = True
    extractDetails(Testfilepath)

    expressions = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}

    files = os.listdir(Framespath)
    rdmnum = random.sample(range(0, len(files)), 9)
    img = []
    for ind in rdmnum:
        img.append(files[ind])

    #detects the face in image using HAAR cascade then crop it then resize it and finally save it.
    face_cascade = cv2.CascadeClassifier('Data/training/haarcascade_frontalface_default.xml') 
    def face_det_crop_resize(img_path):
        img = cv2.imread(img_path)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(img_path)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            face_clip = img[y:y+h, x:x+w]  #cropping the face in image
            cv2.imwrite(img_path, cv2.resize(face_clip, (350, 350)))  #resizing image then saving it

    for image in img:
        face_det_crop_resize(Framespath+'/'+image)


    # Loding images and audio files
    img_data_list=[]
    for image in img:
        input_img=cv2.imread(Framespath + '/'+ image )
        input_img_resize=cv2.resize(input_img,(256,256))
        img_data_list.append(input_img_resize)
    visual_data = np.array(img_data_list)
    visual_data = visual_data.astype('float32')
    visual_data = visual_data/255

    def extract_features(file_name):
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        return mfccsscaled

    features = []
    audio_file = os.listdir(AudioFilepath)[0]
    file_name = str(AudioFilepath)+'/'+str(audio_file)
    features.append(extract_features(file_name))
    audio_data = np.array(features).reshape(-1,40,1)    

    # Loading the saved models
    visual_model = load_model('weights/visual_network.h5')
    audio_model = load_model('weights/audio_network.h5')
    with open('weights/ensemble.pkl', 'rb') as file:
        ensemble_model = pickle.load(file)
    print('Loaded saved models')

    # Predicting on the test data case
    visualpredclass = visual_model.predict_classes(visual_data)
    visualpredprobs = visual_model.predict(visual_data)
    MajClassIndx = np.where(visualpredclass == stats.mode(visualpredclass)[0][0])
    visualpredprob = visualpredprobs[random.choice(MajClassIndx[0])]
    audiopredprob = audio_model.predict(audio_data, verbose=0)

    vdf = pd.DataFrame([visualpredprob],columns=['angerVN','disgustVN','fearVN', 'joyVN', 'neutralVN', 'sadnessVN', 'surpriseVN'])
    adf = pd.DataFrame(audiopredprob,columns=['angerAN','disgustAN','fearAN', 'joyAN', 'neutralAN', 'sadnessAN', 'surpriseAN'])
    frames = [vdf,adf]
    finaldf = pd.concat(frames,axis=1)
    finaldf = finaldf.sample(frac=1).reset_index(drop=True)
    predictedexp = ensemble_model.predict(finaldf)

    for exp, label in expressions.items():
        if label == predictedexp[0]:
            print('Emotion Detected by Ensemble Model:', exp)
if __name__ == '__main__':
    _main()