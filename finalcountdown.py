import numpy as np
import cv2 as cv
import glob


"""
Calculate camera focal length and distortion coefficients based on a set of chessboard images.
"""
def calibrate_camera():
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, the calibration pattern is a 9x6 grid of points
    # i measured the distance between the points to be 31mm
    box_width = 31
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * box_width

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

    return cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

"""
Calculates the camera's world position based on the origin marker (ID 0).
"""
def calculate_camera_world_position(cap, detector, K, dist, marker_length):
    # Accumulate image points and object points for the origin marker
    # accumulating a few frames to get a more accurate estimate
    image_points = []
    object_points = []

    for i in range(10):
        ret, frame = cap.read()
        corners, ids, rejected = detector.detectMarkers(frame)
        origin_marker = int(np.where(ids == 0)[0][0])
        origin_image_points = corners[origin_marker][0]
        image_points.extend(origin_image_points)
        object_points.extend([
            [0, 0, 0],
            [0, 0, -marker_length],
            [-marker_length, 0, -marker_length],
            [-marker_length, 0, 0],
        ])

    origin_image_points = np.array(image_points, dtype=np.float32)
    origin_world_points = np.array(object_points, dtype=np.float32)
    success, rvecs, tvecs = cv.solvePnP(origin_world_points, origin_image_points, K, dist, flags=cv.SOLVEPNP_ITERATIVE)

    np_rodrigues = np.asarray(rvecs[:,:],np.float64)
    rmat = cv.Rodrigues(np_rodrigues)[0]
    camera_world_position = -np.matrix(rmat).T @ np.matrix(tvecs)
    camera_world_position = np.asarray(camera_world_position).reshape(-1)
    return camera_world_position, rmat, rvecs, tvecs

"""
Calculates the world position of a point in a camera image
TODO: we run this 4 times, we can just do a single matrix multiplication instead
"""
def solve_world_position(camera_world, image_points, intrinsics, distortion, rmat):
    undistorted_image_points = cv.undistortPoints(image_points, intrinsics, distortion)
    x, y = undistorted_image_points[0][0]
    Rinv = rmat.T
    ray_im = np.array([x, y, 1])
    ray_world = Rinv @ ray_im

    s = -camera_world[1] / ray_world[1]
    world_point = camera_world + s * ray_world
    return world_point


if __name__ == "__main__":
    ret, K, dist, _, _ = calibrate_camera()
    cap = cv.VideoCapture('output.avi')

    # marker length is 20mm
    marker_length = 20 
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50)
    aruco_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
    camera_world_position, rmat, rvecs, tvecs = calculate_camera_world_position(cap, detector, K, dist, marker_length)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        corners, ids, rejected = detector.detectMarkers(frame)
        cv.aruco.drawDetectedMarkers(frame, corners, ids)


    #    If we dont have 4 corners for the tracking marker, skip drawing the cube
        if len(corners) != 2 or len(corners[1][0]) != 4:
            cv.imshow("img", frame)
            cv.waitKey(1)
            continue

        tracking_marker = int(np.where(ids == 1)[0][0])

        all_cube_world_points = []
        for i in range(4):
            cube_image_points = corners[tracking_marker][0][i]
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
        # Get the first point's coordinates and format them
        first_point = all_cube_world_points[0]
        coord_text = f"({first_point[0]:.1f}, {first_point[1]:.1f}, {first_point[2]:.1f})"
        
        # Calculate text position (slightly offset from the first point)
        text_pos = (projected_points[0][0][0] + 10, projected_points[0][0][1] + 10)
        
        # Add text to the frame
        # Add a black background rectangle for better text visibility
        text_size = cv.getTextSize(coord_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv.rectangle(frame, 
                    (text_pos[0] - 2, text_pos[1] - text_size[1] - 2),
                    (text_pos[0] + text_size[0] + 2, text_pos[1] + 2),
                    (0, 0, 0), -1)
        # Draw text in deep red with increased thickness
        cv.putText(frame, coord_text, text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv.imshow("img", frame)
        # Write the processed frame to the output video
        cv.waitKey(1)

    # Release everything
    cap.release()
    cv.destroyAllWindows()

