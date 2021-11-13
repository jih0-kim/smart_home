# 2nd Week Project KJH LGC

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
app = Flask(__name__)

# AI module
import numpy as np
import cv2
import pickle
import threading

# ETC
from time import sleep, time
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(4,GPIO.OUT,initial = GPIO.LOW)
GPIO.setup(5,GPIO.OUT,initial = GPIO.LOW)
GPIO.setup(14,GPIO.OUT,initial = GPIO.LOW)
GPIO.setup(15,GPIO.OUT,initial = GPIO.LOW)

GPIO.setup(18,GPIO.OUT,initial = GPIO.LOW)
GPIO.setup(27,GPIO.OUT,initial = GPIO.LOW)

PIR_PIN = 7
GPIO_EX.setup(PIR_PIN,GPIO_EX.IN)

LED_1 = 4
LED_2 = 5
LED_3 = 14
LED_4 = 15
LED_list = [LED_1,LED_2,LED_3,LED_4]

LED_Auto_Status = False
FAN_Auto_Status = False
BUZZER_Status = False
AUTO_auto_Status = False

#KEYPAD=======================================
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

#=======================================

is_Obama = 0
message = False
speech_recog = False

# ======================================
# Thread1 - Face Detection

def face_detect():
    global is_Obama
    global face_on_time

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
            roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]

            # recognize? deep learned model predict keras tensorflow pytorch scikit learn
            id_, conf = recognizer.predict(roi_gray)
            if conf>=4 and conf <= 85:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                
                # added code
                if name == 'obama':
                    is_Obama = 1
                    face_on_time = time.time()
                    print(face_on_time)
                    name = ""
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

#=====================================================
#=====================================================
#PIR--------------------------------
pirState = 0
ON = 1
OFF = 0

melodyList_motion = [4]
noteDurations_motion = [1]
motionDetectedTime = 0
motion = 0

def readPir(detect_state):
    global pirState
    global motionDetectedTime
    global motion
    while detect_state:
        input_state = GPIO_EX.input(PIR_PIN)
        if input_state == True:
            if pirState == 0:
                print("\r\nMotion Detected!")
                motionDetectedTime = time.time()
                motion = 1
            pirState = 1
            return 1
        else:
            if pirState == 1:
                print("\r\nMotion Ended")
            pirState = 0
            return 0

# ========================================================================
# speech_recog로 음성 인식 넘어가는 여부 조작
def doubleLock():
    global is_Obama
    global message
    global speech_recog
    global motion
    defult_pw = "1234"
    print(f"time.time() - motionDetectedTime: {time.time() - motionDetectedTime}")
    print(f"motion: {motion}")
    if motion and (time.time() - motionDetectedTime > 3):
        print("It's over 3 sec, buzzer on")
        motion = 0
        if not(is_Obama):
            playBuzzer(melodyList_motion, noteDurations_motion)

    if is_Obama==1 and (not message):
        motion = 0
        displayText("Input Your\nPassword",0,0)
        sleep(2.0)
        lcd.clear()
        initKeypad()
        face_on_time2 =  face_on_time
        # Timelimit after face recognition & three chances to try 
        for i in range(3):
            if (time.time() - face_on_time > 40):
                break
            input_pw = ""
            while(len(input_pw) != 4 and (time.time() - face_on_time2 < 15)):
                keyData = readKeyPad()
                if keyData != -1:
                    input_pw += str(keyData)
                    displayText("Password: " + input_pw,0,0)

            face_on_time2 = time.time()
            
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
    elif is_Obama == -1:
        lcd.clear()
        displayText("ACCESS DENIED",0,0)
        sleep(2.0)
        lcd.clear()
        is_Obama = 0

#=====================================================
#=====================================================

# ====================================================================================================
# Thread2 - Smart Home Control

lcd_rs = digitalio.DigitalInOut(board.D22)
lcd_en = digitalio.DigitalInOut(board.D24)
lcd_d7 = digitalio.DigitalInOut(board.D21)
lcd_d6 = digitalio.DigitalInOut(board.D26)
lcd_d5 = digitalio.DigitalInOut(board.D20)
lcd_d4 = digitalio.DigitalInOut(board.D19)

lcd_columns = 16
lcd_rows = 2

lcd = character_lcd.Character_LCD_Mono(lcd_rs,lcd_en,lcd_d4,lcd_d5,lcd_d6,lcd_d7,lcd_columns,lcd_rows)

#--------------------------------TextLCD----------------------
def initTextlcd():
    lcd.clear()
    lcd.home()
    lcd.cursor_position(0,0)
    sleep(1.0)

def displayText(text = '',col = 8, row = 0):
    lcd.cursor_position(col,row)
    lcd.message = text

def clearTextlcd():
    lcd.clear()
    lcd.message = 'clear LCD\nGoodbye!'
    sleep(2.0)
    lcd.clear()



#--------------------------------Sensor----------------------
#CDS--------------------------------
spi = spidev.SpiDev()
CDS_CHANNEL = 0

def initMcp3208():
    spi.open(0,0)
    spi.max_speed_hz = 100000
    spi.mode = 3

def buildReadCommand(channel):
    startBit = 0x04
    singleEnded = 0x08

    configBit = [startBit | ((singleEnded | (channel & 0x07)) >> 2), (channel & 0x07) << 6, 0x00]
    return configBit

def processAdcValue(result):
    byte2 = (result[1] & 0x0F)
    return (byte2 << 8) | result[2]

def analogRead(channel):
    if (channel > 7) or (channel < 0):
        return -1
    
    r = spi.xfer2(buildReadCommand(channel))
    adc_out = processAdcValue(r)
    return adc_out

def controlMcp3208(channel):
    analogVal = analogRead(channel)
    return analogVal

def readSensor(channel):
    return controlMcp3208(channel)


#온도--------------------------------
dhtDevice = adafruit_dht.DHT11(board.D17,use_pulseio=False)

#GAS--------------------------------

CDS_CHANNEL_GAS = 1
def initMcp3208_gas():
    spi.open(0,0)
    spi.max_speed_hz = 1000000
    spi.mode = 3

def buildReadCommand_gas(channel):
    startBit = 0x04
    singleEnded = 0x08

    configBit = [startBit | ((singleEnded | (channel & 0x07)) >> 2), (channel & 0x07) << 6, 0x00]
    return configBit

def processAdcValue_gas(result):
    byte2 = (result[1] & 0x0F)
    return (byte2 << 8) | result[2]

def analogRead_gas(channel):
    if (channel > 7) or (channel < 0):
        return -1
    
    r = spi.xfer2(buildReadCommand_gas(channel))
    adc_out = processAdcValue_gas(r)
    return adc_out

def controlMcp3208_gas(channel):
    analogVal = analogRead_gas(channel)
    return analogVal

def readSensor_gas(channel):
    return controlMcp3208_gas(channel)


#AUTO--------------------------------
#--------------------------------AUTO LED--------------------------------#

def autoLedON(voltage):
    global LED_Auto_Status
    print(LED_Auto_Status)
    if LED_Auto_Status:
        if voltage < 1:
            GPIO.output(LED_list,GPIO.HIGH)
        elif voltage < 2:
            GPIO.output(LED_list[:2],GPIO.HIGH)
            GPIO.output(LED_list[2:],GPIO.LOW)
        elif voltage < 3:
            GPIO.output(LED_list[:1],GPIO.HIGH)
            GPIO.output(LED_list[1:],GPIO.LOW) 
        else:
            GPIO.output(LED_list,GPIO.LOW)
    else:
        GPIO.output(LED_list,GPIO.LOW)

@app.route('/ledauto/<onoff>')
def turn_autoled(onoff):
    global LED_Auto_Status
    if not speech_recog:
        return "User Not Certificated"
    if onoff:
        if onoff == 'on':
            LED_Auto_Status = True
            while(LED_Auto_Status):
                if voltage < 1:
                    GPIO.output(LED_list,GPIO.HIGH)
                elif voltage < 2:
                    GPIO.output(LED_list[:2],GPIO.HIGH)
                    GPIO.output(LED_list[2:],GPIO.LOW)
                elif voltage < 3:
                    GPIO.output(LED_list[:1],GPIO.HIGH)
                    GPIO.output(LED_list[1:],GPIO.LOW) 
                else:
                    GPIO.output(LED_list,GPIO.LOW) 
            return "LED Auto on"
        
        elif onoff == "off":
            LED_Auto_Status = False
            sleep(0.5)
            GPIO.output(LED_list,GPIO.LOW)
            
            return "LED Auto off"
    return "User Not Certificated"

#--------------------------------AUTO FAN--------------------------------#
@app.route('/fanauto/<onoff>')
def turn_autofan(onoff):
    global FAN_Auto_Status
    if not speech_recog:
        return "User Not Certificated"
    if onoff:
        if onoff == 'on':
            FAN_Auto_Status = True
            while(FAN_Auto_Status):
                if temperature_c>10 or humidity>40:
                    onFan()
                else:
                    offFan()
                
            return "FAN Auto on"
    
        elif onoff == "off":
            FAN_Auto_Status = False
            sleep(0.5)
            offFan()
            return "FAN Auto off"
    return "User Not Certificated"

#--------------------------------AUTO ALL--------------------------------#
@app.route('/allauto/<onoff>')
def turn_autoall(onoff):
    global AUTO_auto_Status
    if not speech_recog:
        return "User Not Certificated"
    if onoff:
        if onoff == 'on':
            AUTO_auto_Status = True
            while(AUTO_auto_Status):
                if temperature_c>10 or humidity>40:
                    onFan()
                else:
                    offFan()
                if voltage < 1:
                    GPIO.output(LED_list,GPIO.HIGH)
                elif voltage < 2:
                    GPIO.output(LED_list[:2],GPIO.HIGH)
                    GPIO.output(LED_list[2:],GPIO.LOW)
                elif voltage < 3:
                    GPIO.output(LED_list[:1],GPIO.HIGH)
                    GPIO.output(LED_list[1:],GPIO.LOW) 
                else:
                    GPIO.output(LED_list,GPIO.LOW)
            return "ALL Auto Working on"
    
        elif onoff == "off":
            AUTO_auto_Status = False
            sleep(0.5)
            GPIO.output(LED_list,GPIO.LOW)
            offFan()
            return "ALL Auto Working off"
    return "User Not Certificated"


# thread 2--------------------------------
def smart_home():
    count = 0  
    initMcp3208()
    global voltage
    global temperature_c
    global humidity
    print("smartHome activate")
    while True:
        try:        
            readVal = readSensor(CDS_CHANNEL)
            voltage = int(readVal * 4.096 / 4096)
            readVal_GAS = readSensor_gas(CDS_CHANNEL)
            voltage_GAS = int(readVal_GAS * 4.096 / 4096)
            motion_ = readPir(ON)
            temperature_c = dhtDevice.temperature
            humidity = dhtDevice.humidity
        
            doubleLock()

            sleep(1)
            print(readVal)
            print(voltage)
            print(motion_)
            print(temperature_c)
            print(humidity)
            print(readVal_GAS)
            print(voltage_GAS)

            cds_print = "voltage: "+str(voltage)
            pir_print = "motion: "+str(motion)
            dht_print = "temp: "+ str(temperature_c)+"\nhumi: "+str(humidity)
            gas_print = "GAS: "+str(voltage_GAS)
            lcd.clear()
            if speech_recog:
                if(count==0):
                    displayText(cds_print,0,0)
                elif(count==1):
                    displayText(pir_print,0,0)
                elif(count==2):   
                    displayText(dht_print,0,0)
                else:
                    displayText(gas_print,0,0)
                    count = 0
                count += 1



        except KeyboardInterrupt:
            lcd.clear()
            # spi.close()
            GPIO.cleanup()
        except RuntimeError as error:
            print(error.args[0])


# =================================================================================================
# Flask Server w/ IFTTT
# PASSIVE

@app.route('/')
def hello():
    return "hello world"
#--------------------------------LED--------------------------------#
@app.route('/led/<passive>')
def ledoff(passive):
    global speech_recog
    if passive:
        if speech_recog:
            if passive == 'on':
                print("LED Turn on")
                GPIO.output(LED_list[:],1)
                return "LED on"
        
            elif passive == "off":
                print('LED Turn off')
                GPIO.output(LED_list[:],0)
                
                return "LED off"
        
    return  "User Not Certificated"

#--------------------------------Fan--------------------------------#
FAN_PIN1 = 18
FAN_PIN2 = 27

def onFan():
    GPIO.output(FAN_PIN1, GPIO.HIGH)
    GPIO.output(FAN_PIN2, GPIO.LOW)

def offFan():
    GPIO.output(FAN_PIN1, GPIO.LOW)
    GPIO.output(FAN_PIN2, GPIO.LOW)

@app.route('/fan/<passive>')
def fanoff(passive):
    global speech_recog
    if passive:
        if speech_recog:
            if passive == 'on':
                print("FAN Turn on")
                onFan()
                
                return "FAN on"
        
            elif passive == "off":
                print('FAN Turn off')
                offFan()
                
                return "FAN off"
    return  "User Not Certificated"



#--------------------------------BUZZER--------------------------------#
BUZZER_PIN = 7
ON = 1
OFF = 0

scale = [261,294,329,349,392,440,493,523]

melodyList = [4,4,5,5,4,4,2,4,4,2,2,1,4,4,5,5,4,4,2,4,2,1,2,0]
noteDurations = [0.5,0.5,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5,1,
                0.5,0.5,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5,1]

GPIO.setup(BUZZER_PIN, GPIO.OUT)
pwm = GPIO.PWM(BUZZER_PIN, 100)
GPIO.setwarnings(False)

def playBuzzer(melodyList, noteDurations):
    pwm.start(100)
    pwm.ChangeDutyCycle(50)
    for i in range(len(melodyList)):
        pwm.ChangeFrequency(scale[melodyList[i]])
        sleep(noteDurations[i])
    pwm.stop()


@app.route('/buzzer/<passive>')
def buzzeroff(passive):
    global BUZZER_Status
    global speech_recog
    if passive:
        if speech_recog:
            if passive == 'on':
                BUZZER_Status = True
                print("buzzer Turn on")   
                while(BUZZER_Status):
                    pwm.start(100)
                    pwm.ChangeDutyCycle(50)
                    for i in range(len(melodyList)):
                        if(BUZZER_Status==False):
                            break
                        pwm.ChangeFrequency(scale[melodyList[i]])
                        sleep(noteDurations[i])
                    pwm.stop()
                return "BUZZER ON"
            elif passive == "off":
                BUZZER_Status = False
                print('buzzer Turn off')
                return "BUZZER off"
        else:
            return "User Not Certificated"



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
