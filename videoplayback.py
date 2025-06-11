import cv2

cap = cv2.VideoCapture('output.avi')

if not cap.isOpened():
    print('Error: Could not open video file.')
    exit()

cv2.namedWindow('Playback')

while True:
    ret, frame = cap.read()
    if not ret:
        print('Reached end of video or failed to read frame.')
        break
    cv2.imshow('Playback', frame)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC key
        print('ESC pressed, exiting playback.')
        break

cap.release()
cv2.destroyAllWindows() 