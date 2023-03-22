#face detection
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
#face detection

#chatbot
import random
import json
import torch
import time

import speech_recognition as sr     #for s2t
import mutagen
from mutagen.mp3 import MP3

from cgitb import text      
from gtts import gTTS       #for t2s
import  os
import math                      

from model import NeuralNet
from nltk_utils import bag_of_words,tokenize
from playsound import playsound
#chatbot

import threading
import time
# import string
import serial

normal = True
angry = False
sad = False
happy = False
person_interacting = None   #who is interacting
engaged = False                        #busy or not

def face_recognition():
    ArduinoSerial=serial.Serial('/dev/ttyACM0',9600,timeout=0.1)
    first_time = False
    global engaged, person_interacting
    video= 0
    modeldir = './model/20180402-114759.pb'
    classifier_filename = './class/classifier.pkl'
    npy='./npy'
    train_img="./train_img"
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
            minsize = 30  # minimum size of face
            threshold = [0.7,0.8,0.8]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            batch_size =100 #1000
            image_size = 182
            input_image_size = 160
            HumanNames = os.listdir(train_img)
            HumanNames.sort()
            print('Loading Model')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile,encoding='latin1')
            
            video_capture = cv2.VideoCapture(video)
            print('Start Recognition')
            while True:
                ret, frame = video_capture.read()
                #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
                cv2.rectangle(frame,(920,500),(1000,580),(255,0,0),2)
                timer =time.time()
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                faceNum = bounding_boxes.shape[0]
                if faceNum == 1:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    for i in range(faceNum):
                        emb_array = np.zeros((1, embedding_size))
                        xmin = int(det[i][0])
                        ymin = int(det[i][1])
                        xmax = int(det[i][2])
                        ymax = int(det[i][3])
                        try:
                            # inner exception
                            if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                print('Face is very close!')
                                continue
                            cropped.append(frame[ymin:ymax, xmin:xmax,:])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                    interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        
                            detected_person = ''
                            if best_class_probabilities>0.87:
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
                                circle_center_x = xmin + math.ceil((xmax-xmin)/2)
                                circle_center_y = ymin+ math.ceil((ymax-ymin)/2)
                                
                                cv2.circle(frame, (circle_center_x,circle_center_y),5,(255,0,0),2)
                                if circle_center_x <920:
                                    ArduinoSerial.write('L'.encode())
                                    time.sleep(0.01)
                                    print("going left")
                                elif circle_center_x>1000:
                                    ArduinoSerial.write('R'.encode())
                                    print("going right")
                                    time.sleep(0.01)
                                else:
                                    ArduinoSerial.write('S'.encode())
                                    time.sleep(0.01)
                                    
                                if circle_center_y < 500:
                                    ArduinoSerial.write('U'.encode())
                                    print("going up")
                                    time.sleep(0.01)
                                elif circle_center_y > 580:
                                    ArduinoSerial.write('D'.encode())
                                    print("going down")
                                    time.sleep(0.01)
                                else:
                                    ArduinoSerial.write('S'.encode())
                                    time.sleep(0.01)
                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]
                                        person_interacting = result_names
                                        # print(person_interacting)
                                        engaged = True
                                        detected_person = best_class_indices
                                        
                                        print("Hello!  [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                    
                                        cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                        cv2.putText(frame, result_names, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)
                                        
                            else :
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                detected_person = 'stranger'
                                person_interacting = 'stranger'
                                if first_time == False:
                                    first_time = True
                                    continue
                                elif first_time == True:
                                    first_time = False
                                    engaged = True
                                cv2.putText(frame, "?", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)
                            print(detected_person)                    
                        except:   
                            
                            print("error")
                        
                endtimer = time.time()
                fps = 1/(endtimer-timer)
                cv2.rectangle(frame,(15,30),(135,60),(0,255,255),-1)
                cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                # cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    video_capture.release()
                    cv2.destroyAllWindows()
                    return

def chatbot():
    global person_interacting,engaged, normal, happy, sad, angry
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('dataset.json','r') as f:
        dataset = json.load(f)
        
    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]



    model = NeuralNet(input_size,hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()


    bot_name = "BuddyBot"
    print("Let's Chat! Type 'Quit' to exit")
    person_interacting_local = None
    while True:
        r = sr.Recognizer()
        text = ""
        if engaged is False:
            continue
        # else:
        #     time.sleep(3)
        if person_interacting_local == None:
            # print(person_interacting+" after engaged false check "+ str(engaged))
            person_interacting_local = person_interacting
        
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("Speak Now...")
            
            audio = r.listen(source)
            
            try:
                text = r.recognize_google(audio)
                print("You have said: \n"+text)
            except Exception as e:
                print("Error: "+ str(e))
        sentence = text
        print("You: "+text)
        abuse_list = ["sucker", "idiot"]
        for i in abuse_list:
            result_abuse = sentence.find(i)
            if result_abuse!=-1:
                normal = False
                angry = True
                out_text = "Please avoid using abusive words and I will not reply to your current sentence because you have used abusive words"
                output = gTTS (text = out_text, lang = "en", slow = False)
                output.save("output.mp3")
                playsound("output.mp3")
                continue
        if sentence.lower() == "quit":
            break
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1,X.shape[0])
        X = torch.from_numpy(X).to(device)
        
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]    #predicted tag
        
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        
        out_text = ""
        
        if prob.item()>0.75:
            for intent in dataset["intents"]:
                if tag == intent["tag"]:
                    out_text = random.choice(intent['responses'])
                    # print(tag)
                    # print("The person is "+ person_interacting_local)
                    if tag == "greeting" or tag == "whoami" and person_interacting_local != "stranger":
                        normal = False
                        happy = True
                    if tag == "greeting" or tag == "whoami":
                        # print("The person is"+ person_interacting_local+" from tag inside")
                        out_text = out_text+ " " + person_interacting_local
                        if tag=="whoami":
                            if person_interacting_local == "stranger":
                                out_text = "Sorry Stranger, I cannot recognize you."
                                normal = False
                                sad = True
                    print(f"{bot_name}: {out_text}")
        else:
            out_text = "Sorry but I couldn't quite catch that"
            print(f'{bot_name}: I do not understand...')
        language = "en"

        output = gTTS (text = out_text, lang = language, slow = False)
        output.save("output.mp3")
        audio = MP3("output.mp3")
        playsound("output.mp3")
        if tag == "goodbye":
            time.sleep(3)
            engaged = False
            person_interacting_local = None
            person_interacting = None
            continue
        # os.system("xdg-open output.mp3")
        # time.sleep(math.ceil(audio.info.length))
        
def facial_expression():
    global normal, happy, sad, angry
    
    while True:
        if normal:
            cap = cv2.VideoCapture("./face_expressions/normal.avi")
            if (cap.isOpened()== False):
                print("Error opening video file")
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    cv2.imshow('Frame', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                else:
                    break
            continue
            
        elif happy:
            happy = False
            normal = True
            cap = cv2.VideoCapture("./face_expressions/happy.avi")
            if (cap.isOpened()== False):
                print("Error opening video file")
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    cv2.imshow('Frame', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                else:
                    break
            continue
            
        elif sad:
            sad = False
            normal = True
            cap = cv2.VideoCapture("./face_expressions/sad.avi")
            if (cap.isOpened()== False):
                print("Error opening video file")
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    cv2.imshow('Frame', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                else:
                    break
            continue
            
        else:
            angry = False
            normal = True
            cap = cv2.VideoCapture("./face_expressions/angry.avi")
            if (cap.isOpened()== False):
                print("Error opening video file")
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    cv2.imshow('Frame', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                else:
                    break
            continue

welcome_text = "Hello, I am buddybot. Nice to meet you. Please wait for a moment while I Process myself after the first boot."
welcome_voice = gTTS (text = welcome_text, lang = "en", slow = False)
welcome_voice.save("welcome_voice.mp3")
playsound("welcome_voice.mp3")

face_thread = threading.Thread(target=face_recognition)
chat_thread = threading.Thread(target=chatbot)
face_exp_thread = threading.Thread(target=facial_expression)

face_exp_thread.start()
face_thread.start()
chat_thread.start()

face_exp_thread.join()
face_thread.join()
chat_thread.join()