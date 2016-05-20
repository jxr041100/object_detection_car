#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <time.h>

#include "CSIM/adas.h"

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );
//bool object_detection_csim(Mat frame);

/** Global variables */
String face_cascade_name = "../Training/face_train/trained_result/cascade7.xml";
CascadeClassifier face_cascade;
bool bLoaded;
bool bLoaded_CModel;


// cvFacedetectParameters param; 

string window_name_OpenCV = "Face detection- OPENCV";
string window_name_CSIM   = "Face detection- SIM";

int main( int argc, const char** argv )
{
  CvCapture* capture;
  capture = cvCaptureFromFile("/home/goddess/workspace/data/carTest2.mp4");

  uint32_t height = 540;
  uint32_t width = 960;

  adas_init(width*0.4, height*0.8);
  Mat frame;
  Mat frameOri;
  Mat gray;
  int keyboard = 0;

  if( capture )
  {
    while( (char)keyboard != 'Q' && (char)keyboard != 'q' && (char)keyboard != 27) 
    {
      keyboard = waitKey( 10 );
      frame =  cvQueryFrame( capture );
      if(frame.empty()) break;

      Size sizeDownscaled(frame.cols/2, frame.rows/2 );
      resize(frame, frame, sizeDownscaled);

      cvtColor( frame, gray, CV_BGR2GRAY );

      Rect roi(width*0.3, height*0.2, width*0.4, height*0.8);
      Mat grayRoi = gray(roi);
      Mat grayRoi2 = grayRoi.clone();

      CarDistance result[20]; 
      uint32_t count = 0;

      clock_t start, end;
      double cpu_time_used;
      start = clock();

      double T1 = (double)cv::getTickCount();

      adas_car_detect(grayRoi2.ptr(0), result, &count);

      double T2 = (double)cv::getTickCount();

      end = clock();
      cpu_time_used = ((double) (end - start));
      //cout << "cpu_time_used csim " << cpu_time_used << endl;
      //cout << "tick difference " << (T2-T1) << endl;
      cout << "Timing: "<< (T2-T1)/cv::getTickFrequency()<<endl;

      for(int i = 0; i<count; i++)
      {
        //printf("$$$$$$$$ x = %d, y = %d, w = %d, h = %d\n",result[i].x,result[i].y,result[i].width,result[i].height);
        CvFont font; 
        //string distance = to_string(result[i].distance);
        //distance = distance + " meters";

        rectangle( frame, Point( result[i].x + width*0.3, result[i].y +  height*0.2), Point( result[i].x+result[i].width + width*0.3, result[i].y+result[i].height +  height*0.2), Scalar( 255, 0, 0 ), 3, 8 );
        //putText(frame, distance, Point( result[i].x + width*0.3+20, result[i].y +  height*0.2+20), FONT_HERSHEY_COMPLEX_SMALL,  1,cvScalar(0,255,0),1, CV_AA);  
        //rectangle( grayRoi, Point( result[i].x , result[i].y), Point( result[i].x+result[i].width , result[i].y+result[i].height ), Scalar( 255, 0, 0 ), 3, 8 );

      }
      imshow("frame",frame);
      //imshow("car" ,grayRoi);
    }
  }
  adas_free();

  int c = waitKey(30);

  //detectAndDisplay( frameOri ); 

  return 0;
}

/**
* @function detectAndDisplay using the opencv lib.
*/

void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat gray;

  if(!bLoaded)
  {
    //-- 1. Load the cascade
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return; };
    bLoaded = true;
  }
  cvtColor( frame, gray, CV_BGR2GRAY );
  //equalizeHist( gray, gray );

  clock_t start, end;
  double cpu_time_used;
  start = clock();

  double T0 = (double)cv::getTickCount();     
  // face_cascade.detectMultiScale( gray, faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30), Size(400, 400) );
  face_cascade.detectMultiScale( gray, faces, 1.2, 10, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30), Size(400, 400));


  double T1 = (double)cv::getTickCount();
  end = clock();
  cpu_time_used = ((double) (end - start));
  cout << "opencv cpu_time_used " << cpu_time_used << endl;
  cout << "tick difference " << (T1-T0) << endl;
  cout << "Timing: "<< (T1-T0)/cv::getTickFrequency()<<endl;
  cout << "Opencv #faces found: " << faces.size() << endl;

  for( int i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    rectangle( frame, Point( faces[i].x, faces[i].y), Point( faces[i].x+faces[i].width, faces[i].y+faces[i].height), Scalar( 0, 0, 0 ), 3, 8 );

    printf("the topx = %d, topy = %d, width =%d,height=%d\n", faces[i].x ,faces[i].y,faces[i].width,faces[i].height);   
  } 
  //-- Show what you got
  imshow( window_name_OpenCV, frame );
  int c = waitKey(30);

}


/**
* @function detectAndDisplay using the Csim code.
*/

/*
bool object_detection_csim(Mat frame)
{


//cout << " object_detection_csim ENTER" << endl;
uint32_t maxDetectedFaceNum = 25;
uint32_t nFace = 0;
Mat gray;

//   cvFacedetectParameters param; 

//color convertion
cvtColor( frame, gray, CV_BGR2GRAY );

//histogram equalization
//equalizeHist( gray, gray );

cvFacedetectResult result[25];  
uint32_t Height = frame.rows;
uint32_t Width = frame.cols;


if(!bLoaded_CModel)
{
//cout << " object_detection_csim 2" << endl;
if(init_face_detection(Height,Width,&param)==false)
{
printf("failed to init face detection \n");
return false;
}
bLoaded_CModel= true;

//cout << param.maxSize  << param.minSize <<endl;

}

//cout << "face_detection_process begin" << endl;

//face detection, completely C sim code here.
double T2 = (double)cv::getTickCount();     
clock_t start, end;
double cpu_time_used;
start = clock();
face_detection_process(gray.ptr(0),Width, Height, Width, &param,  maxDetectedFaceNum, &nFace, result);
double T3 = (double)cv::getTickCount();
end = clock();
cpu_time_used = ((double) (end - start));
cout << "cpu_time_used csin " << cpu_time_used << endl;
cout << "csim tick difference " << (T3-T2) << endl;
cout << ">>> CSIM Object Detection Timing: "<< (T3-T2)/cv::getTickFrequency()<<endl;
//cout <<  "faces found    -- " << nFace << endl;
//push the detected faces into face vector.
for(int i = 0; i<nFace; i++)
{
printf("the topx = %d, topy = %d, width =%d,height=%d\n",result[i].topx,result[i].topy,result[i].width,result[i].height);
rectangle( frame, Point( result[i].topx, result[i].topy), Point( result[i].topx+result[i].width, result[i].topy+result[i].height), Scalar( 255, 0, 0 ), 3, 8 );
}
//-- Show what you got
imshow( window_name_CSIM, frame ); 

//free(param);
//return true;
}
*/