#ifndef OBJ_DETECTOR_API_H
#define OBJ_DETECTOR_API_H
#include "adas.h"

typedef struct
{
    float x;
    float y;
}Point2Df;

typedef struct
{
    uint32_t topx;
    uint32_t topy;
    uint32_t width;
    uint32_t height;
}icvRect;

typedef struct 
{
    uint32_t ROIflag;
    uint32_t topx;
    uint32_t topy;
    uint32_t width;
    uint32_t height;
    uint32_t ID;
}cvFacedetectResult;


typedef struct
{
    uint32_t minSize;
    uint32_t maxSize;
    uint32_t stepSize; 
    float scaleFactor;
}cvFacedetectParameters;


bool object_detection_process( uint8_t * __restrict src, 
      uint32_t srcWidth, 
      uint32_t srcHeight, 
      uint32_t srcStride,
      cvFacedetectParameters *para, 
      uint32_t maxDetectedFaceNum,
      uint32_t *resultFaceNum,
      cvFacedetectResult *result
);
bool init_object_detection(uint32_t Height,uint32_t Width,cvFacedetectParameters *params);
void deinit_object_detection();
#endif
