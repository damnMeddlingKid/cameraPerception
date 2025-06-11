import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import cv2 as cv
import glob

def solve_world_position(camera_world, image_points, intrinsics, distortion, rmat):
    # undistorted_image_points = cv.undistortPoints(image_points, intrinsics, distortion)
    # x, y = undistorted_image_points[0][0]
    #homogenous_image_points = np.hstack((image_points, np.ones((image_points.shape[0], 1))))
    x, y = image_points
    Rinv = rmat.T
    ray_im = np.array([x, y, 1])
    ray_world = np.linalg.inv(intrinsics) @ ray_im
    ray_world = Rinv @ ray_world

    s = -camera_world[1] / ray_world[1]
    world_point = camera_world + s * ray_world
    return world_point

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 31

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob("calibration_data/*.png")

for fname in images:
    print(fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    print(f"{ret} corners in {fname}")

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv.imshow("img", img)
        #cv.waitKey(1500)

ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

cap = cv.VideoCapture('output.avi')
ret, frame = cap.read()
camera_world_position = None
rmat = None
marker_length = 20 # marker length is 20mm
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50)
aruco_params = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
corners, ids, rejected = detector.detectMarkers(frame)

for i in range(9):
    ret, frame = cap.read()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    marker_length = 20 # marker length is 20mm
    corners, ids, rejected = detector.detectMarkers(frame)


    cv.aruco.drawDetectedMarkers(frame, corners, ids)

    """
    Im using the marker with id 0 as the origin of the coordinate system.
    we use this to calculate the camera translation and rotation.
    """

    if camera_world_position is None:
        origin_image_points = corners[0][0]
        origin_world_points = np.array([
            [0, 0, 0],
            [0, 0, -marker_length],
            [-marker_length, 0, -marker_length],
            [-marker_length, 0, 0],
        ], dtype=np.float32)

        success, rvecs, tvecs = cv.solvePnP(origin_world_points, origin_image_points, K, dist, flags=cv.SOLVEPNP_ITERATIVE)

        np_rodrigues = np.asarray(rvecs[:,:],np.float64)
        rmat = cv.Rodrigues(np_rodrigues)[0]
        camera_world_position = -np.matrix(rmat).T @ np.matrix(tvecs)
        camera_world_position = np.asarray(camera_world_position).reshape(-1)

    all_cube_world_points = []

    if len(corners) != 2 or len(corners[1][0]) != 4:
        cv.imshow("img", frame)
        cv.waitKey(1)
        continue

    idx = int(np.where(ids == 1)[0][0])

    for i in range(4):
        cube_image_points = corners[idx][0][i]
        cube_world_points = solve_world_position(camera_world_position, cube_image_points, K, dist, rmat)
        all_cube_world_points.append(cube_world_points)

    # Calculate the top 4 points of the cube by adding marker_length to the z-coordinate
    cube_top_points = []
    for point in all_cube_world_points:
        top_point = point.copy()
        top_point[1] -= marker_length  # Add marker_length to z-coordinate
        cube_top_points.append(top_point)

    # Combine all points for drawing
    all_points = np.array(all_cube_world_points + cube_top_points, dtype=np.float32)

    # Project 3D points to 2D image plane
    projected_points, _ = cv.projectPoints(all_points, rvecs, tvecs, K, dist)
    projected_points = projected_points.astype(np.int32)

    # Draw the cube edges
    # Bottom face
    for i in range(4):
        cv.line(frame, tuple(projected_points[i][0]), tuple(projected_points[(i+1)%4][0]), (0,255,0), 2)
    # Top face
    for i in range(4):
        cv.line(frame, tuple(projected_points[i+4][0]), tuple(projected_points[((i+1)%4)+4][0]), (0,255,0), 2)
    # Vertical edges
    for i in range(4):
        cv.line(frame, tuple(projected_points[i][0]), tuple(projected_points[i+4][0]), (0,255,0), 2)

    cv.imshow("img", frame)
    cv.waitKey(1)




cv.destroyAllWindows()

