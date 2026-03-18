import cv2
import numpy as np

keyboard=np.zeros((1000,1500,3),np.uint8)


def letter(x,y,text):
    #keys
    #x=0
    #y=0
    width=200
    height=200
    thickness=3
    
    cv2.rectangle(keyboard,(x+thickness,y+thickness),(x+width-thickness,y+height-thickness),(255,0,0),3)

    #text settings
    font_letter=cv2.FONT_HERSHEY_PLAIN
    #text='A'
    font_scale=10
    font_thickness=4
    text_size=cv2.getTextSize(text,font_letter,font_scale,font_thickness)[0]
    width_text,height_text= text_size[0],text_size[1]

    text_x=int((width-width_text)/2)+x
    text_y=int((height+height_text)/2)+y

    cv2.putText(keyboard,text,(text_x,text_y),font_letter,font_scale,(255,0,0),font_thickness)

letter(200,0,'B')



cv2.imshow("keyboard",keyboard)
cv2.waitKey(0)
cv2.destroyAllWindows()