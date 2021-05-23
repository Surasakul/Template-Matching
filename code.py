import cv2 
import numpy as np

# load source image 
img = cv2.imread("gg.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread("g.jpg",0)

w = template.shape[1] 
h = template.shape[0]

# select method = CV2. TM_CCOEFF 
res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF)

min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(res)
# for cv2.TM_CCOEFF select Max_loc

top_left = max_loc 
bottom_right = (top_left[0]+w, top_left[1]+h)
# draw rectangle 
cv2.rectangle(img, top_left, bottom_right,(0,255,0),2) # draw BGR color

cv2.imshow("Matching Result",img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
