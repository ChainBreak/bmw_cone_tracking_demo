#! /usr/bin/env python2

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import time
import rospy
import std_msgs.msg

class ConeDetector():
    def __init__(self):
        #setup windows
        cv2.namedWindow("camera",0)
        #cv2.namedWindow("mask",0)
        
        #get capture device
        self.cam = cv2.VideoCapture(1)
        
        #set click callbacks
        cv2.setMouseCallback("camera",self.click_callback)
        
        self.angle_pub = rospy.Publisher("steering_angle",std_msgs.msg.Float32,queue_size=10)
        rospy.init_node("cone_detector")
        
        #constants
        self.smooth_a = 0.5
        self.hist_thresh = 1000
        self.line_color = (0,255,191)
        self.view_slope = 6.5/15.0
        
        #initialise variables
        self.target_x = 0
        self.target_deg = 100.123
        self.test_frame = None
        self.mouse_down = False
        self.mouse_x = -1
        self.mouse_y = -1
        self.edit_timer = time.time()
        
        self.color_mean = np.zeros(3)
        self.color_std = np.zeros(3)
        self.color_low = np.zeros(3)
        self.color_high = np.zeros(3)
        
        try:
            self.color_mean, self.color_std = pickle.load(open("color.pickle","rb"))
        except Exception:
            print("Could not load color")
            
        #loop forever
        try:
            self.loop()
        except KeyboardInterrupt:
            pass
        
        pickle.dump((self.color_mean, self.color_std),open("color.pickle","wb"))
            
        
    def click_callback(self,event,x,y,flags,params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            self.target_x = x
            
        if event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
            
        self.mouse_x = x
        self.mouse_y = y
        
        self.edit_timer = time.time()
        
    
    def loop(self):
        

        while not rospy.is_shutdown():
            ret, self.frame = self.cam.read()
            if not self.frame is None:
                self.test_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2LAB)
                self.update_color()
                
                self.detect_color_object()
         
                self.draw_screen()
                
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
    
    def draw_screen(self):
        img = self.frame
        h,w,c = img.shape
        
        #when the user moves the mouse the edit timer starts
        edit_mode = time.time() - self.edit_timer < 1.0
        
        #if in edit mode, darken everything except mask
        if edit_mode:
            img_dark = cv2.bitwise_and(img,img,mask = cv2.bitwise_not(self.mask))
            img_dark /= 5
            img_cut = cv2.bitwise_and(img,img,mask = self.mask)
            img = cv2.add(img_dark,img_cut)
        else:
            #draw a vertical line along the target
            cv2.line(img,(int(self.target_x),0),(int(self.target_x),img.shape[0]),self.line_color,3)
            
            text       = "%0.0f Deg" % self.target_deg
            font       = cv2.FONT_HERSHEY_PLAIN
            fontScale  = 1
            fontColor  = (0,0,0)
            lineType   = 1
            
            cv2.rectangle(img,(0,h-22),(w,h),self.line_color,-1)
            text_w, text_h = cv2.getTextSize(text,font,fontScale,lineType)[0]
            cv2.putText(img,text,(w/2-text_w/2,h-6),font,fontScale,fontColor,lineType)
            
            
        
        cv2.imshow("camera",img)
        
    def update_color(self):
        h,w,c = self.test_frame.shape
        x_in = self.mouse_x > 0 and self.mouse_x < w
        y_in = self.mouse_y > 0 and self.mouse_y < h
        
        if self.mouse_down and x_in and y_in:
            new_color = self.test_frame[self.mouse_y,self.mouse_x,:].astype(np.float)
            
            self.color_mean += 0.01*(new_color - self.color_mean)
            new_std = np.sqrt((new_color - self.color_mean)**2)
            self.color_std += 0.1*(new_std - self.color_std)
            
        self.color_low = np.clip(self.color_mean - self.color_std*2,0,255).astype(np.uint8)
        self.color_high = np.clip(self.color_mean + self.color_std*2,0,255).astype(np.uint8)

    def detect_color_object(self):
        f_smooth = np.ones(51)/51.0
        
        #get the image shape
        h,w,c = self.test_frame.shape
                
        #get a mask of the pixels that are in range of the collor
        self.mask = cv2.inRange(self.test_frame,self.color_low,self.color_high)
        
        #sum the mask in the y direction to get a histogram down each column of the image
        hist_y = np.sum(self.mask,axis=0)
        
        #Smooth the histogram with a 1d blur filter
        hist_smooth = np.convolve(f_smooth,hist_y,mode="same")
        
        #Calculate the average value of the histogram
        hist_mean = np.mean(hist_smooth)
        
        #Use derivatives to find the peaks of smoothed histogram
        #Calculate the first derivative to get the slope
        hist_d1 = np.convolve( [1.0,0.0,-1.0], hist_smooth, mode="same")
        
        #Threshold the slope so that (+slope,-slope) = (1,0)
        hist_d1 = hist_d1 > 0
        
        #Find slope zero crossing points with second derivative
        hist_d2 = np.convolve( [-1.0,1.0], hist_d1, mode="same")
        
        #Peaks are marked by zero crossing from + to -
        peaks = hist_d2 > 0
        
        #only allow peaks that are taller than the average peak. (removes a little noise)
        peaks *= np.logical_and(hist_smooth > hist_mean, hist_smooth > self.hist_thresh)
        
        #find the indecies of all the peaks
        peak_pos, = np.where(peaks)
        
        if len(peak_pos)>0:
            #Get the value at these peaks
            peak_val = hist_smooth[peak_pos]
            
            #Scale the peaks down by their distance to the current position
            peak_scaled = peak_val*((1 -  abs( (peak_pos - self.target_x) /float(w) ))**2)
            
            #Get the index of highest peak
            i_max = np.argmax(peak_scaled)
            
            #get the position of the highest peak
            self.target_x += self.smooth_a * float(peak_pos[i_max] - self.target_x)
            
            if False:
                plt.cla()
                plt.plot(hist_y)
                plt.plot(hist_smooth)
                plt.plot(peak_pos,peak_val,'o')
                plt.plot(peak_pos,peak_scaled,'o')
                plt.draw()
                plt.pause(0.001)
                
        
        self.target_deg = np.degrees(np.arctan(float(self.target_x - w/2)/(w/2)*self.view_slope))
        
        self.angle_pub.publish(self.target_deg)
                    

if __name__ == "__main__":
    ConeDetector()
