import cv2 
import numpy as np

# load source image 
img = cv2.imread("k.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread("kk.jpg",0)

w = template.shape[1] 
h = template.shape [0]

# select method = cv2.TM_CCOEFF 
res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)

min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res) 
# for cv2.TM_CCOEFF select Max_loc

top_left = max_loc 
bottom_right = (top_left[0]+w, top_left[1]+h)
# draw rectangle 
#cv2.rectangle(img, top_left, bottom_right, (0,255,),2) # draw BGR color

# multi-template matching 
loc = np.where(res >0.9)

for pt in zip (*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0]+w,pt[1]+h),(0,0,255),2)

# cv2.imshow("Peak",res) 
cv2.imshow("Matching Result",img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()