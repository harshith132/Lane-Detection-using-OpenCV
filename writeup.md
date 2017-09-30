

```python
#import all libraries 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
```


```python
#Function for Grey Scale image conversion
def greyscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Function of Canny Edge Detection
def canny(img,low_threshold,high_threshold):
    return cv2.Canny(img,low_threshold, high_threshold)

#Function to filter the grey scale image
def gaussian_blur(img,kernel_size):
    return cv2.GaussianBlur(img,(kernel_size, kernel_size),0)

#Function to determine the region of interest from the Canny image
def reg_of_interest(img,vertices):
    mask=np.zeros_like(img)
    
    if len(img.shape)>2:
        channel_count=img.shape[2]
        ignore_mask_color = (255,)* channel_count
    else:
        ignore_mask_color=255
        
    cv2.fillPoly(mask,vertices, ignore_mask_color)
    
    masked_image=cv2.bitwise_and(img,mask)
    
    return masked_image
```


```python
# Modified Draw Line Function which calculates the slope, averages and extrapolates the lines
def draw_line(img, lines, color=[255, 0, 0], thickness=10):
    imshape = img.shape
    left_x1 = []
    left_x2 = []
    right_x1 = []
    right_x2 = [] 
    y_min = img.shape[0]
    y_max = int(img.shape[0]*0.611)
    
    #Equation for line---> y=mx+b, m=(y-b)/x
    #Calculating the slope and finding the coefficients of line by polyfit function
    for line in lines:
        for x1,y1,x2,y2 in line:
            if ((y2-y1)/(x2-x1)) < 0:     #modified condition to check negative slope
                mc = np.polyfit([x1, x2], [y1, y2], 1)
                left_x1.append(np.int(np.float((y_min - mc[1]))/np.float(mc[0])))
                left_x2.append(np.int(np.float((y_max - mc[1]))/np.float(mc[0])))
            elif ((y2-y1)/(x2-x1)) > 0:
                mc = np.polyfit([x1, x2], [y1, y2], 1)
                right_x1.append(np.int(np.float((y_min - mc[1]))/np.float(mc[0])))
                right_x2.append(np.int(np.float((y_max - mc[1]))/np.float(mc[0])))
                
    #Calculating the average of calculated +/- slopes
    
    l_avg_x1 = np.int(np.nanmean(left_x1))
    l_avg_x2 = np.int(np.nanmean(left_x2))
    r_avg_x1 = np.int(np.nanmean(right_x1))
    r_avg_x2 = np.int(np.nanmean(right_x2))
    
    #Extrapolating the line with considerable thickness
    cv2.line(img, (l_avg_x1, y_min), (l_avg_x2, y_max), color, thickness)
    cv2.line(img, (r_avg_x1, y_min), (r_avg_x2, y_max), color, thickness) 
```


```python
#Hough Transform function
def hough_lines(img,rho,theta,threshold,min_line_length, max_line_gap):
    lines=cv2.HoughLinesP(img,rho,theta,threshold, np.array([]),minLineLength=min_line_length,maxLineGap=max_line_gap)
    line_image=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    draw_line(line_image,lines)
    return line_image

#Weighted Image function
def weighted_image(img,initial_image,alpha=0.8,beta=1,gamma=0):
    return cv2.addWeighted(initial_image, alpha,img,beta,gamma)
```


```python
#Whole Pipeline function 
def pipeline(image):  
    #Parameters for vertices of a 4 sided Polygon
    bot_left = [140, 540]
    bot_right = [980, 540]
    apex_right = [510, 315]
    apex_left = [450, 315]
    
    v = [np.array([bot_left, bot_right, apex_right, apex_left], dtype=np.int32)]
    
    # Run canny edge detection and mask region of interest
    gray = greyscale(image)
    blur = gaussian_blur(gray, 7)
    edge = canny(blur, 50, 125)
    mask = reg_of_interest(edge, v)
    
    ### Run Hough Lines and separate by +/- slope
    lines = cv2.HoughLinesP(mask, 0.8, np.pi/180, 25, np.array([]), minLineLength=50, maxLineGap=200)

    
    
    ### Draw lines and return final image 
    line_img = np.copy((image)*0)
    draw_line(line_img, lines, thickness=10)
    
    line_img = reg_of_interest(line_img, v)
    final = weighted_image(line_img, image)
    


    return final
```


```python
from moviepy.editor import VideoFileClip, ImageClip
from IPython.display import HTML

def process_image(image):
    result = pipeline(image)
    return result

white_output = 'test_videos_output/whiteSolidRight.mp4'
clip1 = VideoFileClip('test_videos/solidWhiteRight.mp4')
white_clip = clip1.fl_image(process_image)
%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video test_videos_output/whiteSolidRight.mp4
    [MoviePy] Writing video test_videos_output/whiteSolidRight.mp4
    

    100%|███████████████████████████████████████████████████████████████████████████████▋| 221/222 [00:06<00:00, 36.75it/s]
    

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/whiteSolidRight.mp4 
    
    Wall time: 6.76 s
    


```python
from moviepy.editor import VideoFileClip, ImageClip
from IPython.display import HTML

def process_image(image):
    result = pipeline(image)
    return result

white_output = 'test_videos_output/YellowSolidLeft.mp4'
clip1 = VideoFileClip('test_videos/solidYellowLeft.mp4')
white_clip = clip1.fl_image(process_image)
%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video test_videos_output/YellowSolidLeft.mp4
    [MoviePy] Writing video test_videos_output/YellowSolidLeft.mp4
    

    100%|███████████████████████████████████████████████████████████████████████████████▉| 681/682 [00:19<00:00, 35.10it/s]
    

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/YellowSolidLeft.mp4 
    
    Wall time: 20.1 s
    


```python
image=mpimg.imread("test_images/solidWhiteCurve.jpg")
result=pipeline(image)
plt.imshow(result)
plt.show()
```


![png](output_7_0.png)



```python
image=mpimg.imread("test_images/solidWhiteRight.jpg")
result=pipeline(image)
plt.imshow(result)
plt.show()
```


![png](output_8_0.png)



```python
image=mpimg.imread("test_images/solidYellowCurve.jpg")
result=pipeline(image)
plt.imshow(result)
plt.show()
```


![png](output_9_0.png)



```python
image=mpimg.imread("test_images/solidYellowCurve2.jpg")
result=pipeline(image)
plt.imshow(result)
plt.show()
```


![png](output_10_0.png)



```python
image=mpimg.imread("test_images/solidYellowLeft.jpg")
result=pipeline(image)
plt.imshow(result)
plt.show()
```


![png](output_11_0.png)



```python
image=mpimg.imread("test_images/whiteCarLaneSwitch.jpg")
result=pipeline(image)
plt.imshow(result)
plt.show()
```


![png](output_12_0.png)


# **Finding Lane Lines on the Road**  
 
## Write-up  
 
 
--- 
 
**Finding Lane Lines on the Road** 
 
The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road 
* Reflection on my work, potential short comings and Suggestions 
 
 
 
--- 
 
### Reflection 
 
This project involves Lane Detection using Python and OpenCV. The important concepts which are used to detect lanes are:
*        Grey-Scaling an Image using cvtColor()
*        Edge Detection using Canny Edge Detection Algorithm [Canny() function] 
*        Filtering the Image by using Gaussian Blur [GaussianBlur() function] . 
*        Region Masking using fillPoly(). 
*        Converting an Image Space into a parameter space using Hough Transform [HoughLinesP()]. 
*        Drawing solid lines on the Edge Image using drawLines() . 

The image captured is converted into a Grey Scale and then Canny Detection Algorithm is applied to denote the edges in the image. It determines the edge w.r.t sudden pixel density variations. So, the lower and upper threshold values have to be selected appropriately and I have chosen it to be 50 and 125 respectively. Suitable kernel size has to be determined to filter the edge detected image which should be an odd number (7 in this case). I have selected a 4 sided Polygon for region masking. Then the image has to be transformed into the parameter space using Hough Transform. All the edges are converted to short Line segments by selecting the parameters of Hough transform such as rho, theta, minimum Line length and maximum line gap. All other spaces in the image except the lanes are masked and the lanes are drawn solid lines (red color) by calculating the positive and negative slopes. The positive and negative slopes are checked and their respective slopes are calculated. The averages of all coordinates are calculated and extrapolated with a red line on both lanes by deciding on the coordinates of the polygon and coefficients of the line which is found out using polyfit(). The whole procedure of drawing line can be found in draw_line() function. 

 
### 2.Potential shortcomings with my current pipeline 
 
 
One potential shortcoming would be when the road is down the slope or inclined, the parameters of the region masking is not dynamic enough while drawing the line. Another shortcoming would be during night time when there is very less intensity of light, I guess the images cannot be processed properly.  
 
 
### 3. Possible improvements to my pipeline 
 
A possible improvement would be to consider the curvature of the road before finding lanes and another would be to find where to drive on roads which do not have lanes. 
 
 
















```python

```
