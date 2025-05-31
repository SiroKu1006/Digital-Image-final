import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('drop.avi')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("data/frame%d.bmp" % count, image)     # save frame as BMP file
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  count += 1
