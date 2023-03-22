import cv2 as cv
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from ultralytics import YOLO
from sort import *
import math
import numpy as np
import os


import uuid
import kivy
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.app import App
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.graphics.texture import Texture



fd = FaceDetector(minDetectionCon=0.8)


model = YOLO('yolov8n.pt')
tracker = Sort(max_age=20,min_hits=3)
line = [80,250,250,250]

lines = [330,200,500,260]
counterin = []
counterout = []
num_of_people = 100
classnames = []
with open('classes.txt','r') as f:
    classnames = f.read().splitlines()


print(classnames[0])
class UoBGrp14App(App):
    def build(self):
        # color sequence is RGB for the first 3 digits
        Window.clearcolor = (1, 1, 0, 1)
        # Window.size = (1920,1080)
        # self.icon = (r'C:\Users\NANOR\Desktop\pictures\htulogo.png')

        self.cam_img = Image(size_hint=(1, 1))
        self.label = Label(text='Human Counting', size_hint=(1, 0.1), font_size=40,
                           italic=0, bold=1,color=(1,0,0,1))


        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.label)
        layout.add_widget(self.cam_img)
        # in case you are using webcam uncomment the line below
        # self.cap = cv.VideoCapture(0)
        self.cap = cv.VideoCapture('vid.mp4')
        Clock.schedule_interval(self.frame, 1.0 / 33.0)
        return layout

    def frame(self, *args):
        _, img = self.cap.read()
        print(img.shape)

        # Yolo object detection
        detections = np.empty((0, 5))

        results = model(img, stream=True)
        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_detect = box.cls[0]

                class_detect = int(class_detect)
                class_detect = classnames[class_detect]
                conf = math.ceil(confidence * 100)

                if class_detect == 'person'  and conf > 60:
                    x1, y1,x2, y2 = int(x1), int(y1), int(x2), int(y2)



                    current_detections = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_detections))

        # Counting part
        total = []
        tracker_result = tracker.update(detections)

        cv.line(img,(line[0],line[1]),(line[2],line[3]),(255,255,255),1)
        cv.line(img, (lines[0], lines[1]), (lines[2], lines[3]), (255, 255, 255), 1)

        for track_result in tracker_result:
            x1, y1, x2, y2, id = track_result
            x1, y1, x2, y2,id  = int(x1), int(y1), int(x2), int(y2),int(id)
            total.append(id)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            # cv.circle(img,(cx,cy),8,(0,255,255),-1)
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(img, f'{id}', [x1 + 8, y1 - 12],
                               scale=2, thickness=2)


            if line[0] < cx < line[2] and line[1] - 15 < cy < line[1] + 15:
                cv.line(img, (line[0], line[1]), (line[2], line[3]), (255, 255, 255), 1)
                if counterin.count(id) == 0:
                    counterin.append(id)

            if lines[0] < cx < lines[2] and lines[1] - 30 < cy < lines[1] + 30:
                cv.line(img, (lines[0], lines[1]), (lines[2], lines[3]), (255, 255, 255), 1)
                if counterout.count(id) == 0:
                    counterout.append(id)


        cvzone.putTextRect(img, f'IN = {len(counterin)}', [10, 34], thickness=4, scale=2.3, border=2)
        cvzone.putTextRect(img, f'SEE ={len(total)}', [330, 34], thickness=4, scale=2.3, border=2)
        cvzone.putTextRect(img, f'OUT ={len(counterout)}', [690, 34], thickness=4, scale=2.3, border=2)

        val = np.interp(len(total), [0, num_of_people], [470, 50])
        cv.putText(img, 'P', (24, 303), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 4)
        cv.rectangle(img, (8, 310), (50, 470), (0, 0, 255), 5)
        cv.rectangle(img, (8, int(val)), (50, 470), (255, 0, 255), -1)



        buf = cv.flip(img, 0).tostring()
        img_texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.cam_img.texture = img_texture




if __name__ == "__main__":
    UoBGrp14App().run()
