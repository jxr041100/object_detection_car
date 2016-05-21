#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <time.h>

#include "CSIM/adas.h"
#include "CSIM/CompressiveTracker.h"
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
	VideoCapture cap("/home/goddess/workspace/data/carTest2.mp4"); 
	//capture = cvCaptureFromFile("../data/car.mp4");
	uint32_t height;
	uint32_t width;
	Mat frame;
	Mat frameOri;
	Mat gray;
	int keyboard = 0;
    int frame_number = 0;
    bool bInitial = false;
    // CT framework
	CompressiveTracker ct;
    int previous_car_size = -1;
    bool bUpdateTracking = false;
    Rect box;
   
	
		while(1) 
		{
			bool success = cap.read(frame); 
            if (!success){
                cout << "Cannot read  frame " << endl;
                break;
            }
            Size sizeDownscaled(frame.cols/2, frame.rows/2 );
			resize(frame, frame, sizeDownscaled);

            if(!bInitial)
            {
                height = frame.rows;
                width = frame.cols;
                adas_init(width*0.4, height*0.8);
                bInitial = true;
            }		

			cvtColor( frame, gray, CV_BGR2GRAY );
		

            double cpu_time_used;
            double T1 = (double)cv::getTickCount();
#if 1
            if(frame_number % 30 == 0)
            {	
                Rect roi(width*0.3, height*0.2, width*0.4, height*0.8);
                Mat grayRoi = gray(roi);
                Mat grayRoi2 = grayRoi.clone();
			    CarDistance result[20];	
			    uint32_t count = 0; 	   
			    adas_car_detect(grayRoi2.ptr(0), result, &count);
			               
                int curr_max_car_size = 0;
			    for(int i = 0; i<count; i++)
			    {
				    //printf("x = %d, y = %d, w = %d, h = %d\n",result[i].x,result[i].y,result[i].width,result[i].height);
				    CvFont font; 
				    //string distance = to_string(result[i].distance);
				    //distance = distance + " meters";

				    rectangle( frame, Point( result[i].x + width*0.3, result[i].y +  height*0.2), Point( result[i].x+result[i].width + width*0.3, result[i].y+result[i].height +  height*0.2), Scalar( 0, 0, 255 ), 3, 8 );
				    //putText(frame, distance, Point( result[i].x + width*0.3+20, result[i].y +  height*0.2+20), FONT_HERSHEY_COMPLEX_SMALL,	1,cvScalar(0,255,0),1, CV_AA);  
				    //rectangle( grayRoi, Point( result[i].x , result[i].y), Point( result[i].x+result[i].width , result[i].y+result[i].height ), Scalar( 255, 0, 0 ), 3, 8 );

                    //pick up the largest car for tracking.
                    if(result[i].width > curr_max_car_size)
                    {
                        box.x = result[i].x + width*0.3;
                        box.y = result[i].y +  height*0.2;
                        box.width = result[i].width;
                        box.height = result[i].height;
                        curr_max_car_size = result[i].width;
                        bUpdateTracking = true;
                        printf("result[%d].width = %d",i, result[i].width);
                    }                   
                }
                cout << " detecting processing frame id " << frame_number << endl;
            }
            else
            {                
                if(bUpdateTracking == true)
                {
                    ct.init(gray, box);
                    bUpdateTracking = false;
                    previous_car_size = box.width;
                }
                if(box.width >0)
                {
                    ct.processFrame(gray,box);
                    cout << " Tracking processing frame id " << frame_number << endl;                   
                }
            }
#endif
            double T2 = (double)cv::getTickCount();     
            //cout << "tick difference " << (T2-T1) << endl;
            cout << "Timing: "<< (T2-T1)/cv::getTickFrequency()<<endl;
            cout << "box.x  = "<< box.x << " box.y = " << box.y << "box.width = " << box.width << "box.height = " << box.height << endl;
            if(box.width>0)
                rectangle( frame, Point( box.x , box.y), Point(box.x + box.width, box.y + box.height), Scalar( 255, 0, 0 ), 3, 8 );
		  
            frame_number++;
            imshow("frame",frame);
            int c = waitKey(30);
		}
	adas_free();
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