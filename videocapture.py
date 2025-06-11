import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

recording = False
out = None

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        if not recording:
            # Start recording
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = 20.0
            frame_size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter('output.avi', fourcc, fps, frame_size)
            recording = True
            print("Started recording...")
    if recording and out is not None:
        out.write(frame)

cam.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
