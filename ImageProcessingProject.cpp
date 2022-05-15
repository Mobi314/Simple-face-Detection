/*******************************************************************************
 * Name        : ImageProcessingProject.cpp
 * Author      : Tahrim Imon, Janet Hamrani, Enis Rama
 * Version     : 1.0
 * Date        : 5/14/2022
 * Description : Face Detection using OpenCV Library
 * Pledge      : I pledge my honor that I have abided by the Stevens Honor System
 ******************************************************************************/

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp> 
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void faceDetection(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale)
{
    vector<Rect> faces1, faces2;
    Mat grayScale, sm_img;
    double fix = 1 / scale;

    /*cvtColor function to convert the image to Grey Scale*/
    cvtColor(img, grayScale, COLOR_BGR2GRAY);

    /*Uniformally resize the Greyscale Image*/
    resize(grayScale, sm_img, Size(), fix, fix, INTER_LINEAR);
    equalizeHist(sm_img, sm_img);

    /*Using the object detection library, we can detect faces
    Using the cascade classifier*/
    cascade.detectMultiScale(sm_img, faces1, 1.1,
        2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    //Draw circles around the faces
    for(size_t i = 0; i < faces1.size(); i++){
        
        vector<Rect> nestedObjects;
        Point center;

        Rect rect = faces1[i];
        Mat sm_imgROI;
        
        Scalar color = Scalar(0, 255, 0); //Scalar Function Draws the color Green
        int radius;

        double aspect_ratio = (double)rect.width / rect.height;
        
        /*Draws circle around identified face
        using circle function within a specified aspect ratio constraint*/
        if(0.25 < aspect_ratio && aspect_ratio < 2){
            center.x = cvRound((rect.x + rect.width * 0.5) * scale);
            center.y = cvRound((rect.y + rect.height * 0.5) * scale);
            radius = cvRound((rect.width + rect.height) * 0.25 * scale);
            circle(img, center, radius, color, 3, 8, 0);
        }
        else
            rectangle(img, cv::Point(cvRound(rect.x * scale), cvRound(rect.y * scale)),
                cv::Point(cvRound((rect.x + rect.width - 1) * scale),
                    cvRound((rect.y + rect.height - 1) * scale)), color, 3, 8, 0);
        if(nestedCascade.empty())
            continue;
        sm_imgROI = sm_img(rect);

        // Detection of eyes int the input image
        nestedCascade.detectMultiScale(sm_imgROI, nestedObjects, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        /*Like the facial draw, we draw circles around the eyes*/
        for(size_t j = 0; j < nestedObjects.size(); j++){
            Rect nr = nestedObjects[j];
            center.x = cvRound((rect.x + nr.x + nr.width * 0.5) * scale);
            center.y = cvRound((rect.y + nr.y + nr.height * 0.5) * scale);
            radius = cvRound((nr.width + nr.height) * 0.25 * scale);
            circle(img, center, radius, color, 3, 8, 0);
        }
    }

    // Show Processed Image with detected faces
    imshow("Face Detection", img);
}

int main(int argc, const char** argv)
{
    /*VideoCapture class for capturing the frames in which
    the faces will be detected and displayed*/
    VideoCapture cap;
    Mat frm, img;

    // PreDefined trained XML classifiers with facial features
    CascadeClassifier cascade, nestedCascade;
    double sc = 1;

    // Load classifiers from "opencv/data/haarcascades" directory 
    nestedCascade.load("C:/OpenCV-4.5.5/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");

    // Change path before execution 
    cascade.load("C:/OpenCV-4.5.5/opencv/sources/data/haarcascades/haarcascade_frontalcatface.xml");

    // Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video
    cap.open(0);
    if (cap.isOpened())
    {
        // Capture frames from video and detect faces
        cout << "Face Detection Started...." << endl;
        while (1){
            cap >> frm;
            if (frm.empty())
                break;
            
            Mat frame1 = frm.clone();
            faceDetection(frame1, cascade, nestedCascade, sc);
            
            char keyPress = (char)waitKey(10);

            // Press q to exit from window
            if (keyPress == 27 || keyPress == 'q' || keyPress == 'Q')
                break;
        }
    }
    else
        cout << "Could not Open Camera";
    return 0;
}
