import cv2
import numpy as np

import time

from traceback2 import print_tb
from utils import detector, predictor, getFacelandmarks468
from utils import warpTriangles, getFacelandmarks, cropFacemask
from utils import delaunayTriangulation, getTriangleIndex, adjustColorToneofFace

def swapFaces(indexes_triangles, landmarks_points,landmarks_points2, img ,img2_new_face):
    
    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Triangulation of second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        warped_triangle = warpTriangles(points, points2, cropped_triangle, cropped_tr2_mask, rect2)
        
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
    
    return img2_new_face


def main():
    
    img = cv2.imread("bradley_cooper.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    cap = cv2.VideoCapture(0)

    landmark_points  = getFacelandmarks468(img)
    
    if (landmark_points is not None):

        face_crop_mask1 = cropFacemask(landmark_points, img)

        triangle_points = delaunayTriangulation(landmark_points)
        
        indexes_triangles = getTriangleIndex(triangle_points, landmark_points)
    
        while True:
            
            ret, img2 = cap.read()
                        
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            swapped_new_face = np.zeros_like(img2)
             
            landmark_points2 = getFacelandmarks468(img2)
                    
            swapped_new_face = swapFaces(indexes_triangles, landmark_points,
                                                            landmark_points2, img, swapped_new_face )
            if swapped_new_face is not None:

                seamlessclone = adjustColorToneofFace(img2_gray, img2, swapped_new_face, landmark_points2 )
                    
                cv2.imshow("face swap", seamlessclone )
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

        cap.release()

if __name__ == '__main__':

    main()
