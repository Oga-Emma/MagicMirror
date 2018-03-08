package ng.com.mrseven.magicmirror.faceswap;

/**
 * Created by Mr. Seven on 3/5/2018.
 */

import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_RGB2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.convexHull;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.fillConvexPoly;
import static org.bytedeco.javacpp.opencv_imgproc.fillPoly;
import static org.bytedeco.javacpp.opencv_imgproc.warpAffine;

public class FaceSwap {

//   /* class FaceSwap(object):
//    def __init__(self, hairmodel_image, selfie_image):
//            """
//            :param hairmodel_image: a numpy array of the primary image ie we use hair from image1.
//        :param selfie_image: a numpy array representation of an image. we use face from image2, face from image2 is replaced by face from image1
//        """
//
//    self.hair_model = hairmodel_image
//    self.selfie_image = selfie_image  # cv2.cvtColor(selfie_image, cv2.COLOR_RGB2GRAY) # selfie_image
//    self.align_points = list(range(0, 68))

    private Mat hairModel;
    private Mat selfieImage;
    private int[] alignPoints = new int[68];

    public FaceSwap(Mat hairModel, Mat selfieImage){

        this.hairModel = hairModel;
        this.selfieImage = selfieImage;

        for(int i = 0; i < alignPoints.length; i++){
            alignPoints[i] = i;
        }
    }
//
//    @staticmethod
//    def get_landmarks(image):
//            """
//            :param: image, a numpy array representation of the input image to find landmarks
//        :type: numpy array
//        :return: landmarks, a (68 x 2) matrix of coordinates, of special features in any image, for this instance a face.
//        :type: matrix of dimension (68 x 2)
//        """
//    image_ = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
//    detector = dlib.get_frontal_face_detector()
//    rects = detector(image_, 1)
//    predictor = dlib.shape_predictor(os.path.dirname(__file__)+"/shape_predictor_68_face_landmarks.dat")
//
//    landmarks = np.matrix([[int(p.x), int(p.y)] for p in predictor(image_, rects[0]).parts()])
//
//            return landmarks

    private void getLandmarks(Mat image){
        Mat tempImage = new Mat();

        cvtColor(image, tempImage, COLOR_RGB2GRAY);
        //detector = dlib.get_frontal_face_detector()
        //rects = detector(image_, 1)
        //predictor = dlib.shape_predictor(os.path.dirname(__file__)+"/shape_predictor_68_face_landmarks.dat")

        //landmarks = np.matrix([[int(p.x), int(p.y)] for p in predictor(image_, rects[0]).parts()])

        //return landmarks

    }
//
//    def draw_convex_hull(self, image, points):
//            """
//            :param image: numpy array on which to draw the convex hull (a convex polyhedral).
//            :param points: coordinates of the vertices of the convex hull/ convex polyhedral to be drawn on the input image.
//        :return: numpy array, with convex hull overlaid on the input image.
//        """
//    points = cv2.convexHull(points)
//            cv2.fillConvexPoly(image, points, color=1)

    private Mat drawConvexHull(Mat image, Point points){
//
//        Mat tempImage = new Mat();
//
//        convexHull(new Mat(points), tempImage);
//        fillConvexPoly(image, , 1, new Scalar(1));

        return null;
    }
//
//    @staticmethod
//    def get_facemask(image, landmarks):
//            """
//            :param image, numpy array
//        :param landmarks: matrix, default = (68x2) collection of coordinates of primary features of a face: eyes, mouth, ...
//            :return: matrix with same shape as image, a rescaled version of the input image.
//            """
//            # image_ = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
//    face_mask = np.zeros(image.shape, dtype=image.dtype)
//    face_convex_hull = np.array(cv2.convexHull(landmarks))[:, 0]
//    face_convex_hull = face_convex_hull.astype(np.int32)
//    face_mask = cv2.fillPoly(face_mask, [face_convex_hull], (255, 255, 255))
//            # face_mask = numpy.array([face_mask, face_mask, face_mask]).transpose((1, 2, 0))
//            return face_mask

    public void getFaceMask(Mat image){
        Mat faceMask = new Mat();
        Mat faceConvexHull = new Mat();
        PointerPointer faseConvexHull;
        Mat faceMak;


    }
//
//    @staticmethod
//    def transformation_from_points(points1, points2):
//            """
//    Return an affine transformation [s * r | T] such that:
//    sum ||s*r*p1,i + T - p2,i||^2
//    is minimized.
//            """
//            # Solve the procrustes problem by subtracting centroids, scaling by the
//        # standard deviation, and then using the SVD to calculate the rotation.
//
//    points1 = points1.astype(np.float64)
//    points2 = points2.astype(np.float64)
//
//    c1 = np.mean(points1, axis=0)
//    c2 = np.mean(points2, axis=0)
//    points1 -= c1
//    points2 -= c2
//
//            s1 = np.std(points1)
//    s2 = np.std(points2)
//    points1 /= s1
//    points2 /= s2
//
//            u, s, vt = np.linalg.svd(points1.T * points2)
//
//        # The r we seek is in fact the transpose of the one given by u * vt. This
//        # is because the above formulation assumes the matrix goes on the right
//        # (with row vectors) where as our solution requires the matrix to be on the
//        # left (with column vectors).
//    r = (u * vt).T
//
//        return np.vstack([np.hstack(((s2 / s1) * r, c2.T - (s2 / s1) * r * c1.T)), np.matrix([0., 0., 1.])])

    public static void transformationFromPoints(Point points1, Point points2){

    }
//
//    @staticmethod
//    def warp(image, m, dshape):
//    output_image = np.zeros(dshape, dtype=image.dtype)  # this creates a black background .
//            cv2.warpAffine(image, m[:2], (dshape[1], dshape[0]), dst=output_image, borderMode=cv2.BORDER_TRANSPARENT,
//    flags=cv2.WARP_INVERSE_MAP)
//            return output_image

    public static void warp(Mat image, int[] dshape){
        Mat outputImage = new Mat();

//        warpAffine(image, outputImage, 2, (dshape[1]), (dshape[0]),BORDER_TRANSPARENT);
    }
//
//    @staticmethod
//    def face_center(hairface_landmarks):
//            """
//            :param hairface_landmarks: np array, each row contains [x,y] co-ordinates of landmarks on the input image
//        :return: (center-x, center-y) co-ordinates of the center of the image.
//        """
//    hairface_convex_hull = cv2.convexHull(hairface_landmarks)[:, 0]
//
//    rect_vertices = cv2.boundingRect(np.float32(hairface_convex_hull))
//    center = (int(rect_vertices[0]) + int(rect_vertices[2] / 2), int(rect_vertices[1]) + int(rect_vertices[3] / 2))
//
//            return center
//
//    def edgeRefinement(self, source_image, destination_image, sourceface_mask, hairface_landmarks):
//            """
//            :param source_image: np.array representation of the selfie model image
//        :param destination_image: np.array representation of the hairstyle model (adjusted) image
//        :param sourceface_mask: np.array (black and white image of the selfie model's face )
//            :param hairface_landmarks: np.array 2d matrix of the hairmodel facial landmarks.
//            :return: np.array image of the seamlessly cloned face from the source_image on the template face at the destination
//    image.
//        """
//    center = self.face_center(hairface_landmarks)
//    print("center", center)
//    edge_blended = cv2.seamlessClone(source_image, destination_image, sourceface_mask, center, cv2.NORMAL_CLONE)
//    print("eb:", edge_blended.shape)
//        return edge_blended
//
//    def swap(self):
//            """
//
//            :return: np.array. The selfie model with a new hairstyle from the hairmodel.
//            """
//    print("here")
//    hairface_landmarks = self.get_landmarks(self.hair_model)  # hair
//            user_face_landmarks = self.get_landmarks(self.selfie_image)  # face
//    print("here")
//    procrustes = self.transformation_from_points(hairface_landmarks[self.align_points], user_face_landmarks[self.align_points])
//    warped_selfie_image = self.warp(self.selfie_image, procrustes, self.hair_model.shape)  # self.sourceface_image.shape)
//    sourceface_landmarks = self.get_landmarks(warped_selfie_image)  # face
//
//        # obtain the landmark parameters as a list.
//            p17 = tuple(sourceface_landmarks[17].tolist()[0])
//    p19 = tuple(sourceface_landmarks[19].tolist()[0])
//    p24 = tuple(sourceface_landmarks[24].tolist()[0])
//    p26 = tuple(sourceface_landmarks[26].tolist()[0])
//    p8 = tuple(sourceface_landmarks[8].tolist()[0])
//
//            # Calculate the distance btw (p8, p24)
//    r8_24 = np.linalg.norm(np.matrix(p8) - np.matrix(p24))
//            # Calculate the distance btw (p8, p17)
//    r8_17 = np.linalg.norm(np.matrix(p8) - np.matrix(p17))
//            # Calculate the distance btw (p6, p26)
//    r8_26 = np.linalg.norm(np.matrix(p8) - np.matrix(p26))
//            # Calculate the distance btw (p6, p19)
//    r8_19 = np.linalg.norm(np.matrix(p8) - np.matrix(p19))
//
//            # Calculate the translation/increment.
//
//        # Euclidean distance from p8 to p17
//    r1 = r8_17 * (1 + 1/32)
//            # Euclidean distance from p8 to p68
//    r2 = r8_19 * (1 + 1/8)
//            # Euclidean distance from p6 to p69
//    r3 = r8_24 * (1 + 1/8)
//            # Euclidean distance from p6 to p70
//    r4 = r8_26 * (1 + 1/32)
//
//            # Angle between line 10,17 and x-axis
//            p8x, p8y = p8
//    p17x, p17y = p17
//            theta1 = np.math.atan(abs((p17y - p8y) / (p17x - p8x)))
//
//        # Angle between line 6,19 and x-axis
//            p19x, p19y = p19
//    theta2 = np.math.atan(abs((p19y - p8y) / (p19x - p8x)))
//
//            # Angle between line 6,24 and x-axis
//            p24x, p24y = p24
//    theta3 = np.math.atan(abs((p24y - p8y) / (p24x - p8x)))
//
//            # Angle between line 6,26 and x-axis
//            p26x, p26y = p26
//    theta4 = np.math.atan(abs((p26y - p8y) / (p26x - p8x)))
//
//    p68 = (int(p8x - r2 * np.cos(theta2)), int(p8y - r2 * np.sin(theta2)))
//    p69 = (int(p8x + r3 * np.cos(theta3)), int(p8y - r3 * np.sin(theta3)))
//    p70 = (int(p8x - r1 * np.cos(theta1)), int(p8y - r1 * np.sin(theta1)))
//    p71 = (int(p8x + r4 * np.cos(theta4)), int(p8y - r4 * np.sin(theta4)))
//
//    sourceface_landmarks_aug = sourceface_landmarks.copy()
//    sourceface_landmarks_aug[19, 0] = p68[0]
//    sourceface_landmarks_aug[19, 1] = p68[1]
//    sourceface_landmarks_aug[24, 0] = p69[0]
//    sourceface_landmarks_aug[24, 1] = p69[1]
//    sourceface_landmarks_aug[17, 0] = p70[0]
//    sourceface_landmarks_aug[17, 1] = p70[1]
//    sourceface_landmarks_aug[26, 0] = p71[0]
//    sourceface_landmarks_aug[26, 1] = p71[1]
//
//    combined_facemask = self.get_facemask(warped_selfie_image, sourceface_landmarks_aug)
//    print(warped_selfie_image.shape)
//    print(self.hair_model.shape)
//    print(combined_facemask.shape)
//    newlook = self.edgeRefinement(warped_selfie_image, self.hair_model, combined_facemask, sourceface_landmarks_aug)
//            # re-shape newlook
//        # newlook_landmark = np.array(self.get_landmarks(newlook))
//            # xmax, ymax = newlook_landmark.max(axis=0)
//            # newlook = newlook[0:int(ymax), 0:int(newlook.shape[1])]
//    newlook = cv2.putText(newlook, "MagicMirror.ai", (int((1/20) * newlook.shape[1]), int((1-1/32)*newlook.shape[0])), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2.0, (182, 130, 29), 4)  # bgr rgb(29,130,182)
//
//        return newlook
//
//
//    from faceShape import getFaceShape
//import datetime as dt
//    from PIL import Image
//
//            base_path = os.path.dirname(__file__)
//    test_date = dt.datetime.now()
//    save_path = base_path+'/file_storage/trash/test_'+str(test_date)+'.jpg'
//    img2_path = "IMG_20171219_035105.jpg"   #  "file_storage/trash/heart/white/heart_ehurley2.jpg"  # [
//    img1_path = "mimiee.jpg"  # {"file_storage/trash/heart/dark/5.jpg"  # ["file_storage/trash/round/chocolate/round_jhudson_031.jpg"
//        img1 = Image.open(img1_path)
//        img2 = Image.open(img2_path)




}
