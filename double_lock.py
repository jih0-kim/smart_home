# python3 exercise4.py
# ./ngrok http 5000
# googlesamples-assistant-pushtotalk

# if display error:
# export DISPLAY=:0.0
# xhost +local:root
# xhost +localhost

# smarthome module
import RPi.GPIO as GPIO
import GPIO_EX
import adafruit_dht
import board
import digitalio
import adafruit_character_lcd.character_lcd as character_lcd
import spidev

# flask module
from flask import Flask

# AI module
import numpy as np
import cv2
import pickle
import threading

# ETC
from time import sleep, time

# =================================================================================================
# Thread1 - Face Detection

is_Obama = 0
message = False
speech_recog = False

def face_detect():
    global is_Obama

    face_cascade = cv2.CascadeClassifier('/home/pi/opencv/OpenCV-Python-Series/src/cascades/data/haarcascade_frontalface_alt2.xml')
    # eye_cascade = cv2.CascadeClassifier('/home/pi/opencv/OpenCV-Python-Series/src/cascades/data/haarcascade_eye.xml')
    # smile_cascade = cv2.CascadeClassifier('/home/pi/opencv/OpenCV-Python-Series/src/cascades/data/haarcascade_smile.xml')


    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("/home/pi/opencv/OpenCV-Python-Series/src/recognizers/face-trainner.yml")

    labels = {"person_name": 1}
    with open("/home/pi/opencv/OpenCV-Python-Series/src/pickles/face-labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            #print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]

            # recognize? deep learned model predict keras tensorflow pytorch scikit learn
            id_, conf = recognizer.predict(roi_gray)
            if conf>=4 and conf <= 85:
                #print(5: #id_)
                #print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                
                # added code
                if name == 'obama':
                    is_Obama = 1
                elif name == 'clinton':
                    is_Obama = -1
                else:
                    is_Obama = 0
                # added code - end

            img_item = "7.png"
            cv2.imwrite(img_item, roi_color)

            color = (255, 0, 0) #BGR 0-255 
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
            #subitems = smile_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in subitems:
            #   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



# ====================================================================================================
# Thread2 - Smart Home Control

# LED
LED_1 = 4
LED_2 = 5
LED_3 = 14
LED_4 = 15
LEDs = [LED_1, LED_2, LED_4, LED_3]

# flag on flask request
led_on = False

# led
def turn_led():
    global led_on
    if led_on:
        GPIO.output(LEDs, 1)
    else:
        GPIO.output(LEDs, 0)

# keypad
ROW0_PIN = 0
ROW1_PIN = 1
ROW2_PIN = 2
ROW3_PIN = 3
COL0_PIN = 4
COL1_PIN = 5
COL2_PIN = 6

COL_NUM = 3
ROW_NUM = 4

g_preData = 0

colTable = [COL0_PIN,COL1_PIN,COL2_PIN]
rowTable = [ROW0_PIN,ROW1_PIN,ROW2_PIN,ROW3_PIN]

def initKeypad():
    for i in range(0,COL_NUM):
        GPIO_EX.setup(colTable[i], GPIO_EX.IN)
    for i in range(0,ROW_NUM):
        GPIO_EX.setup(rowTable[i], GPIO_EX.OUT)

def selectRow(rowNum):
    for i in range(0,ROW_NUM):
        if rowNum == (i+1):
            GPIO_EX.output(rowTable[i], GPIO_EX.HIGH)
            sleep(0.001)
        else:
            GPIO_EX.output(rowTable[i], GPIO_EX.LOW)
            sleep(0.001)
    return rowNum

def readCol():
    keypadstate = -1
    for i in range(0,COL_NUM):
        inputKey = GPIO_EX.input(colTable[i])
        if inputKey:
            keypadstate = keypadstate + (i+2)
            sleep(0.5)
    return keypadstate

def readKeyPad():
    global g_preData
    keyData = -1

    runningStep = selectRow(1)    
    row1Data = readCol()          
    sleep(0.001)
    if (row1Data != -1):
        keyData = row1Data
    
    for i in range(1,ROW_NUM):
        if (runningStep == i):
            if keyData == -1:
                runningStep = selectRow(i+1)
                row2Data = readCol()
                sleep(0.001)
                if (row2Data != -1):
                    keyData = row2Data + i*3
    sleep(0.1) 

    if keyData == -1:
        return -1
    elif keyData == 10:
        keyData = '*'
    elif keyData == 11:
        keyData = 0
    elif keyData == 12:
        keyData = '#'
    
    if g_preData == keyData:
        g_preData = -1
        return -1
    g_preData = keyData

    print("\r\nKeypad Data: %s" %keyData)
    return keyData

# LCD screen
lcd_rs = digitalio.DigitalInOut(board.D22)
lcd_en = digitalio.DigitalInOut(board.D24)
lcd_d7 = digitalio.DigitalInOut(board.D21)
lcd_d6 = digitalio.DigitalInOut(board.D26)
lcd_d5 = digitalio.DigitalInOut(board.D20)
lcd_d4 = digitalio.DigitalInOut(board.D19)

lcd_columns = 16
lcd_rows = 2
lcd = character_lcd.Character_LCD_Mono(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7, 
                                        lcd_columns, lcd_rows)

def initTextlcd():
    lcd.clear()
    lcd.home()
    lcd.cursor_position(0,0)
    sleep(1.0)

def displayText(text='', col=0, row=0):
    lcd.cursor_position(col,row)
    lcd.message = text

def clearTextlcd():
    lcd.clear()
    lcd.message = 'clear LCD\nGoodbye!'
    sleep(2.0)
    lcd.clear()

# ========================================================================
# speech_recog로 음성 인식 넘어가는 여부 조작
def doubleLock():
    global is_Obama
    global message
    global speech_recog
    defult_pw = "1234"

    if is_Obama==1 and (not message):
        displayText("Input Your\nPassword",0,0)
        sleep(2.0)
        lcd.clear()
        initKeypad()
        
        for i in range(3):
            input_pw = ""
            while(len(input_pw) != 4):
                keyData = readKeyPad()
                if keyData != -1:
                    input_pw += str(keyData)
                    displayText("Password: " + input_pw,0,0)
            
            if defult_pw == input_pw:
                message = True
                speech_recog = True
            else:
                message = False
            displayText("CORRECT" if message else "FAILED",0,1)
            sleep(2.0)
            lcd.clear()
            if message: 
                break
            displayText("Try Again.\nYou Tried "+str(i+1)+" time",0,0)
            sleep(2.0)
            lcd.clear()  
        is_Obama = 0
        message = False
    elif is_Obama==-1:
        lcd.clear()
        displayText("ACCESS DENIED",0,0)
        sleep(2.0)
        lcd.clear()
        is_Obama = 0


# message랑 is_Obama를 false 만들면 조작이 안 되니까 새 변수 만들어야 할 듯

# thread에서는 global variance를 못 쓴다. 매우 충격적. 
# 함수에서 호출해서 불러 쓰고 그 함수를 스레드 안에 넣을 것.
# thread 2
def smart_home():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(LEDs, GPIO.OUT, initial=GPIO.LOW)

    print("smartHome activate")

    while True:
        try:            
            # on-off by flags
            turn_led()
            doubleLock()

        except KeyboardInterrupt:
            lcd.clear()
            spi.close()
            GPIO.cleanup()
        except RuntimeError as error:
            print(error.args[0])

# =================================================================================================
# Flask Server w/ IFTTT

app = Flask(__name__)

@app.route('/')
def hello():
    return "hello world"


@app.route('/LED/<onoff>')
def led_onoff(onoff):
    global led_on

    if onoff == "on":
        led_on = True

    elif onoff == "off":
        led_on = False

    return "led"

# =================================================================================================
# execute code

if __name__ == "__main__":

    # run faceDetact function with thread t1
    global t1
    t1 = threading.Thread(target=face_detect)
    t1.daemon = True
    t1.start()
    
    # run smarthome function with thread t1
    global t2
    t2 = threading.Thread(target=smart_home)
    t2.daemon = True
    t2.start()

    app.run(host='0.0.0.0', port=5000, debug=False)
