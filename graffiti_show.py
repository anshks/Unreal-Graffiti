import numpy as np
import cv2 as cv

bBlob = cv.imread('blueblob.png',-1) # -1 loads with transparency to get the fourth channel
gBlob = cv.imread('greenblob.png',-1)
rBlob = cv.imread('redblob.png',-1)
yBlob = cv.imread('yellowblob.png',-1)
crosshair=cv.imread('crosshair.png',-1)

en = 1

def overlay_transparent(background_img, img_to_overlay_t, overlay_size, x, y):

	bg_img = background_img.copy()
	err  = bg_img.copy()
	try:
		height = bg_img.shape[0]
		width = bg_img.shape[1]
		
		img_to_overlay_t = cv.resize(img_to_overlay_t.copy(), overlay_size)
		print(img_to_overlay_t.shape)
		# Extract the alpha mask of the RGBA image, convert to RGB
		b,g,r,a = cv.split(img_to_overlay_t)
		overlay_color = cv.merge((b,g,r))

		# Apply some simple filtering to remove edge noise using median blur
		mask = cv.medianBlur(a,5)
		h, w, _ = overlay_color.shape
		roi = bg_img[int(y):int(y+h), int(x):int(x+w)]
		mask = cv.resize(mask,(roi.shape[1],roi.shape[0]))

		mask = mask.astype(np.uint8)
		roi = roi.astype(np.uint8)

		# Black-out the area behind the logo in our original ROI
		img1_bg = cv.bitwise_and(roi.copy(),roi.copy(),mask = cv.bitwise_not(mask))

		# Mask out the logo from the logo image.
		img2_fg = cv.bitwise_and(overlay_color,overlay_color,mask = mask)

		# Update the original image with our new ROI
		# print(img1_bg.shape,img2_fg.shape)
		bg_img[int(y):int(y+h), int(x):int(x+w)] = cv.add(img1_bg, img2_fg,dtype=cv.CV_8U)
		return bg_img
	except:
		return err

# Define the upper and lower boundaries for a color to be considered "Blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# Setup the Paint interface
paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv.rectangle(paintWindow, (160,1), (255,65), (255, 0, 0), -1)
paintWindow = cv.rectangle(paintWindow, (275,1), (370,65), (0, 255, 0), -1)
paintWindow = cv.rectangle(paintWindow, (390,1), (485,65), (0, 0, 255), -1)
paintWindow = cv.rectangle(paintWindow, (505,1), (600,65), (0, 255, 255), -1)
cv.putText(paintWindow, "CLEAR ALL", (49, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv.LINE_AA)
cv.putText(paintWindow, "BLUE", (185, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
cv.putText(paintWindow, "GREEN", (298, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
cv.putText(paintWindow, "RED", (420, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
cv.putText(paintWindow, "YELLOW", (520, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv.LINE_AA)

cv.namedWindow('Paint', cv.WINDOW_NORMAL)

#Load the video
camera = cv.VideoCapture(0)

color = 0

tempWindow=paintWindow

# Keep looping
while True:
	paintWindow=tempWindow
	# Grab the current paintWindow
	(grabbed, frame) = camera.read()
	frame = cv.flip(frame, 1)
	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

	# Add the coloring options to the frame
	frame = cv.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
	frame = cv.rectangle(frame, (160,1), (255,65), (255, 0, 0), -1)
	frame = cv.rectangle(frame, (275,1), (370,65), (0, 255, 0), -1)
	frame = cv.rectangle(frame, (390,1), (485,65), (0, 0, 255), -1)
	frame = cv.rectangle(frame, (505,1), (600,65), (0, 255, 255), -1)
	cv.putText(frame, "CLEAR ALL", (49, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
	cv.putText(frame, "BLUE", (185, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
	cv.putText(frame, "GREEN", (298, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
	cv.putText(frame, "RED", (420, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
	cv.putText(frame, "YELLOW", (520, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv.LINE_AA)

	# Check to see if we have reached the end of the video
	if not grabbed:
		break

	# Determine which pixels fall within the blue boundaries and then blur the binary image
	blueMask = cv.inRange(hsv, blueLower, blueUpper)
	blueMask = cv.erode(blueMask, kernel, iterations=2)
	blueMask = cv.morphologyEx(blueMask, cv.MORPH_OPEN, kernel)
	blueMask = cv.dilate(blueMask, kernel, iterations=1)

	# Find contours in the image
	(_, cnts, _) = cv.findContours(blueMask.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
	center = None

	# Check to see if any contours were found
	if len(cnts) > 0:
		# Sort the contours and find the largest one -- we
		# will assume this contour correspondes to the area of the bottle cap
		cnt = sorted(cnts, key = cv.contourArea, reverse = True)[0]
		# Get the radius of the enclosing circle around the found contour
		((x, y), radius) = cv.minEnclosingCircle(cnt)
		# Draw the circle around the contour
		cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
		# Get the moments to calculate the center of the contour (in this case Circle)
		M = cv.moments(cnt)
		center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
		cx = center[0]
		cy = center[1]

		if center[1] <= 65:
			if 40 <= center[0] <= 140: # Clear All
				paintWindow[67:,:,:] = 255
			elif 160 <= center[0] <= 255:
				color = 0 # Blue
				paintWindow = overlay_transparent(paintWindow, bBlob, (55,55), cx-2.5, cy+2.5)
				frame = overlay_transparent(frame, bBlob, (55,55), cx-2.5, cy+2.5)
			elif 275 <= center[0] <= 370:
				color = 1 # Green
				paintWindow = overlay_transparent(paintWindow, gBlob, (55,55), cx-2.5, cy+2.5)
				frame = overlay_transparent(frame, gBlob, (55,55), cx-2.5, cy+2.5)
			elif 390 <= center[0] <= 485:
				color = 2 # Red
				paintWindow = overlay_transparent(paintWindow, rBlob, (55,55), cx-2.5, cy+2.5)
				frame = overlay_transparent(frame, rBlob, (55,55), cx-2.5, cy+2.5)
			elif 505 <= center[0] <= 600:
				color = 3 # Yellow
				paintWindow = overlay_transparent(paintWindow, yBlob, (55,55), cx-2.5, cy+2.5)
				frame = overlay_transparent(frame, yBlob, (55,55), cx-2.5, cy+2.5)
		else:
			if color == 0:
				if en == 1:
					paintWindow = overlay_transparent(paintWindow, bBlob, (55,55), cx-2.5, cy+2.5) # Blue
					tempWindow = paintWindow
				elif en == 0:
					tempWindow=paintWindow
					paintWindow = overlay_transparent(paintWindow, crosshair, (55,55), cx-2.5, cy+2.5)
				#frame = overlay_transparent(frame, bBlob, (55,55), cx-2.5, cy+2.5)

			elif color == 1:
				if en == 1:
					paintWindow = overlay_transparent(paintWindow, gBlob, (55,55), cx-2.5, cy+2.5) # Green
					tempWindow = paintWindow
				elif en == 0:
					tempWindow=paintWindow
					paintWindow = overlay_transparent(paintWindow, crosshair, (55,55), cx-2.5, cy+2.5)
				#frame = overlay_transparent(frame, gBlob, (55,55), cx-2.5, cy+2.5)
			
			elif color == 2:
				if en == 1:
					paintWindow = overlay_transparent(paintWindow, rBlob, (55,55), cx-2.5, cy+2.5) # Red
					tempWindow = paintWindow
				elif en == 0:
					tempWindow = paintWindow
					paintWindow = overlay_transparent(paintWindow, crosshair, (55,55), cx-2.5, cy+2.5)
				#frame = overlay_transparent(frame, rBlob, (55,55), cx-2.5, cy+2.5)

			elif color == 3:
				if en == 1:
					paintWindow = overlay_transparent(paintWindow, yBlob, (55,55), cx-2.5, cy+2.5) # Yellow
					tempWindow = paintWindow
				elif en == 0:
					tempWindow = paintWindow
					paintWindow = overlay_transparent(paintWindow, crosshair, (55,55), cx-2.5, cy+2.5)	
				#frame = overlay_transparent(frame, yBlob, (55,55), cx-2.5, cy+2.5)

	# Show the frame and the paintWindow image
	window_to_be_printed=paintWindow.copy()
	newX,newY = window_to_be_printed.shape[1]*2.8, window_to_be_printed.shape[0]*2
	window_to_be_printed = cv.resize(window_to_be_printed,(int(newX),int(newY)))
	cv.imshow("Tracking", frame)
	cv.imshow("Paint", window_to_be_printed)

	# If the 'q' key is pressed, stop the loop
	if cv.waitKey(1) & 0xFF == ord("q"):
		break

# Cleanup the camera and close any open windows
cv.imwrite("final_graffiti.png", paintWindow)
camera.release()
cv.destroyAllWindows()