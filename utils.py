import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def warpTriangles(points, points2, cropped_triangle, cropped_tr2_mask, rect2):
    
    # Warp triangles
    (x, y, w, h) = rect2
    points = np.float32(points)
    points2 = np.float32(points2)
    M = cv2.getAffineTransform(points, points2)
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h), flags=cv2.INTER_NEAREST)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
    
    return warped_triangle


def adjustColorToneofFace(img2_gray, img2, img2_new_face, landmark_points2 ):
    
    landmark_points2_np = np.array(landmark_points2, np.int32)
    convexhull2 = cv2.convexHull(landmark_points2_np)
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)


    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)

    return seamlessclone

def getFacelandmarks(image_gray):
    faces = detector(image_gray)

    if (len(faces)):

        for face in faces:
            landmarks = predictor(image_gray, face)
            landmarks_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))
                cv2.circle(image_gray, (x, y), 3, (0, 0, 255), -1)

        return landmarks_points
    
    return None


def cropFacemask(landmark_points, img):
    
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    points = np.array(landmark_points, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 255)
    face_image_1 = cv2.bitwise_and(img, img, mask=mask)

    return face_image_1


def delaunayTriangulation(landmark_points):
    
    landmark_points_np = np.array(landmark_points, np.int32)
    convexhull = cv2.convexHull(landmark_points_np)
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_points)
    triangles = subdiv.getTriangleList()
    triangle_points = np.array(triangles, dtype=np.int32)
    
    return triangle_points


def extractIndexNparray(nparray):

    return nparray[0][0]


def getTriangleIndex(triangle_points, landmark_points):
    landmark_points = np.array(landmark_points, np.int32)
    indexes_triangles = []
    for t in triangle_points:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((landmark_points == pt1).all(axis=1))
        index_pt1 = extractIndexNparray(index_pt1)

        index_pt2 = np.where((landmark_points == pt2).all(axis=1))
        index_pt2 = extractIndexNparray(index_pt2)

        index_pt3 = np.where((landmark_points == pt3).all(axis=1))
        index_pt3 = extractIndexNparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
    
    return  indexes_triangles