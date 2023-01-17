/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include "gstnvdsmeta.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)


static const int NUM_CLASSES_YOLO = 80;
static bool DICT_LPR_READY=false;
std::vector<std::string> DICT_LPR;
static bool DICT_VMN_READY=false;
std::vector<std::string> DICT_VMN;

void *set_metadata_ptr(std::array<float, 10> & arr)
{
    gfloat *user_metadata = (gfloat*)g_malloc0(10*sizeof(gfloat));

    for(int i = 0; i < 10; i++) {
       user_metadata[i] = arr[i];
    }
    return (void *)user_metadata;
}

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

struct ObjectPoint{
   float ctx;
   float cty;
   float width;
   float height;
   float confidence;
   int classId;
};

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomRLP(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYolor(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV4TLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C" bool NvDsInferParseCustomYoloV4LPR (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString);


/* YOLOv4 implementations */
static NvDsInferParseObjectInfo convertBBoxYoloV4(const float& bx1, const float& by1, const float& bx2,
                                     const float& by2, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution

    float x1 = bx1 * netW;
    float y1 = by1 * netH;
    float x2 = bx2 * netW;
    float y2 = by2 * netH;

    x1 = clamp(x1, 0, netW-1);
    y1 = clamp(y1, 0, netH-1);
    x2 = clamp(x2, 0, netW-1);
    y2 = clamp(y2, 0, netH-1);

    b.left = x1;
    b.width = clamp(x2 - x1, 0, netW-1);
    b.top = y1;
    b.height = clamp(y2 - y1, 0, netH-1);
    // std::cout << " left " << b.left << " width " << b.width << " top " << b.top << " height " << b.height << std::endl;
    return b;
}

static NvDsInferParseObjectInfo convertBBoxRLP(const float& bx1, const float& by1, const float& bx2,
                                     const float& by2, const uint& netW, const uint& netH, 
                                     const float& lmx1, const float& lmy1,
                                     const float& lmx2, const float& lmy2,
                                     const float& lmx3, const float& lmy3,
                                     const float& lmx4, const float& lmy4,
                                     const float& lmx5, const float& lmy5
                                     )
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution

    float x1 = bx1 * netW;
    float y1 = by1 * netH;
    float x2 = bx2 * netW;
    float y2 = by2 * netH;

    x1 = clamp(x1, 0, netW-1);
    y1 = clamp(y1, 0, netH-1);
    x2 = clamp(x2, 0, netW-1);
    y2 = clamp(y2, 0, netH-1);

    b.left = x1;
    b.width = clamp(x2 - x1, 0, netW-1);
    b.top = y1;
    b.height = clamp(y2 - y1, 0, netH-1);
    std::cout << " LP: left " << b.left << " width " << b.width << " top " << b.top << " height " << b.height << std::endl;
    return b;
}



/* YOLOR implementations */
static NvDsInferParseObjectInfo convertBBoxYolor(const float& bx1, const float& by1, const float& bx2,
                                     const float& by2, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution

    float x1 = clamp(bx1, 0, netW-1);
    float y1 = clamp(by1, 0, netH-1);
    float x2 = clamp(bx2, 0, netW-1);
    float y2 = clamp(by2, 0, netH-1);

    b.left = x1;
    b.width = clamp(x2 - x1, 0, netW-1);
    b.top = y1;
    b.height = clamp(y2 - y1, 0, netH-1);
    // std::cout << "Vehicle: left " << b.left << " width " << b.width << " top " << b.top << " height " << b.height << std::endl;
    return b;
}


static void addBBoxProposalYoloV4(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxYoloV4(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static void addBBoxProposalRLP(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, 
                     const float& lmx1, const float& lmy1,
                     const float& lmx2, const float& lmy2,
                     const float& lmx3, const float& lmy3,
                     const float& lmx4, const float& lmy4,
                     const float& lmx5, const float& lmy5,
                     const int maxIndex, const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxRLP(bx, by, bw, bh, netW, netH, lmx1, lmy1, lmx2, lmy2, lmx3, lmy3, lmx4, lmy4, lmx5, lmy5);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static void addBBoxProposalYolor(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxYolor(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}


static std::vector<NvDsInferParseObjectInfo> decodeYoloV4Tensor(
    const float* boxes, const float* scores, const float* classes,
    const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
    const uint& netW, const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;


    uint bbox_location = 0;
    uint score_location = 0;
    for (uint b = 0; b < num_bboxes; ++b)
    {
        float bx1 = boxes[bbox_location];
        float by1 = boxes[bbox_location + 1];
        float bx2 = boxes[bbox_location + 2];
        float by2 = boxes[bbox_location + 3];
        float maxProb = scores[score_location];
        int maxIndex = (int) classes[score_location];

        if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
        {
            addBBoxProposalYoloV4(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
        }

        bbox_location += 4;
        score_location += 1;
    }

    return binfo;
}

static std::vector<NvDsInferParseObjectInfo> decodeRLPTensor(
    const float* boxes, const float* scores, const float* classes,
    const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
    const uint& netW, const uint& netH, const float* landmarks)
{
    std::vector<NvDsInferParseObjectInfo> binfo;


    uint bbox_location = 0;
    uint score_location = 0;
    uint landmark_location = 0;
    for (uint b = 0; b < num_bboxes; ++b)
    {
        float bx1 = boxes[bbox_location];
        float by1 = boxes[bbox_location + 1];
        float bx2 = boxes[bbox_location + 2];
        float by2 = boxes[bbox_location + 3];
        float maxProb = scores[score_location];
        int maxIndex = (int) classes[score_location];
        float lmx1 = landmarks[landmark_location];
        float lmy1 = landmarks[landmark_location + 1];
        float lmx2 = landmarks[landmark_location + 2];
        float lmy2 = landmarks[landmark_location + 3];
        float lmx3 = landmarks[landmark_location + 4];
        float lmy3 = landmarks[landmark_location + 5];
        float lmx4 = landmarks[landmark_location + 6];
        float lmy4 = landmarks[landmark_location + 7];
        float lmx5 = landmarks[landmark_location + 8];
        float lmy5 = landmarks[landmark_location + 9];

        if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
        {
            addBBoxProposalRLP(bx1, by1, bx2, by2, netW, netH, lmx1, lmy1, lmx2, lmy2, lmx3, lmy3, lmx4, lmy4, lmx5, lmy5, maxIndex, maxProb, binfo);
        }

        bbox_location += 4;
        score_location += 1;
        landmark_location += 10;
    }

    return binfo;
}

static std::vector<NvDsInferParseObjectInfo> decodeYolorTensor(
    const float* boxes, const float* scores, const float* classes,
    const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
    const uint& netW, const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;


    uint bbox_location = 0;
    uint score_location = 0;
    for (uint b = 0; b < num_bboxes; ++b)
    {
        float bx1 = boxes[bbox_location];
        float by1 = boxes[bbox_location + 1];
        float bx2 = boxes[bbox_location + 2];
        float by2 = boxes[bbox_location + 3];
        float maxProb = scores[score_location];
        int maxIndex = (int) classes[score_location];

        if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
        {
            addBBoxProposalYolor(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
        }

        bbox_location += 4;
        score_location += 1;
    }

    return binfo;
}

// static std::vector<NvDsInferParseObjectInfo> decodeSCRFDTensor(
//     const float* boxes, const float* scores, const float* classes,
//     const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
//     const uint& netW, const uint& netH)
// {
//     std::vector<NvDsInferParseObjectInfo> binfo;


//     uint bbox_location = 0;
//     uint score_location = 0;
//     const float half_margin = 22.0;
//     for (uint b = 0; b < num_bboxes; ++b)
//     {
//         float bx1 = boxes[bbox_location] - half_margin;
//         float by1 = boxes[bbox_location + 1] - half_margin;
//         float bx2 = boxes[bbox_location + 2] + half_margin;
//         float by2 = boxes[bbox_location + 3] + half_margin;
//         float maxProb = scores[score_location];
//         int maxIndex = (int) classes[score_location];

//         if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
//         {
//             addBBoxProposalYolor(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
//             // std::cout << "SCRFD: " << bx1 << " " << by1 << " " << bx2 << " " << by2 << " " << " " << netW << " " << " " << netH << " " << std::endl;
//         }

//         bbox_location += 4;
//         score_location += 1;
//     }

//     return binfo;
// }

bool compareObject(ObjectPoint obj1, ObjectPoint obj2)
{
    return (obj1.ctx < obj2.ctx);
}

static std::string mergeDetectionResult(const std::vector<ObjectPoint> &objectList){
    //Divide to 2 lines
    std::vector <ObjectPoint> objectList1;
    std::vector <ObjectPoint> objectList2;
    const float max_dis = 0.3;
    float min_y = 1.;
    for (ObjectPoint obj : objectList){
        if (obj.cty < min_y)
            {
                min_y =  obj.cty;
            }
    }
    for (ObjectPoint obj : objectList){
        if (obj.cty - min_y < max_dis)
            {
                objectList1.push_back(obj);
            }
        else objectList2.push_back(obj);
    } 
    // Sort each line
    std::string licensePlate = "";
    if (objectList1.size() > 0){
        sort(objectList1.begin(), objectList1.end(), compareObject);
        for (ObjectPoint obj: objectList1){
            licensePlate += DICT_LPR[obj.classId];
            }
        }
    if (objectList2.size() > 0){
        sort(objectList2.begin(), objectList2.end(), compareObject);
        for (ObjectPoint obj: objectList2){
            licensePlate += DICT_LPR[obj.classId];
            }
        }

    return licensePlate;

}

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    // {
    //     std::cerr << "WARNING: Num classes mismatch. Configured:"
    //               << detectionParams.numClassesConfigured
    //               << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    // }

    std::vector<NvDsInferParseObjectInfo> objects;
    const NvDsInferLayerInfo &n_bboxes   = outputLayersInfo[0];
    const NvDsInferLayerInfo &boxes      = outputLayersInfo[1]; // (num_boxes, 4)
    const NvDsInferLayerInfo &scores     = outputLayersInfo[2]; // (num_boxes, )
    const NvDsInferLayerInfo &classes    = outputLayersInfo[3]; // (num_boxes, )
    

    int num_bboxes = *(const int*)(n_bboxes.buffer);


    assert(boxes.inferDims.numDims == 2);
    assert(scores.inferDims.numDims == 1);
    assert(classes.inferDims.numDims == 1);

    // std::cout << "Network Info: " << networkInfo.height << "  " << networkInfo.width << std::endl;

    std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeYoloV4Tensor(
            (const float*)(boxes.buffer), (const float*)(scores.buffer), (const float*)(classes.buffer), num_bboxes, detectionParams,
            networkInfo.width, networkInfo.height);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}


extern "C" bool NvDsInferParseCustomRLP(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    // {
    //     std::cerr << "WARNING: Num classes mismatch. Configured:"
    //               << detectionParams.numClassesConfigured
    //               << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    // }

    std::vector<NvDsInferParseObjectInfo> objects;
    const NvDsInferLayerInfo &n_bboxes   = outputLayersInfo[0];
    const NvDsInferLayerInfo &boxes      = outputLayersInfo[1]; // (num_boxes, 4)
    const NvDsInferLayerInfo &scores     = outputLayersInfo[2]; // (num_boxes, )
    const NvDsInferLayerInfo &classes    = outputLayersInfo[3]; // (num_boxes, )
    const NvDsInferLayerInfo &landmarks  = outputLayersInfo[4]; // (num_boxes, 10)
    

    int num_bboxes = *(const int*)(n_bboxes.buffer);


    assert(boxes.inferDims.numDims == 2);
    assert(scores.inferDims.numDims == 1);
    assert(classes.inferDims.numDims == 1);
    assert(landmarks.inferDims.numDims == 2);

    // std::cout << "Network Info: " << networkInfo.height << "  " << networkInfo.width << std::endl;

    std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeRLPTensor(
            (const float*)(boxes.buffer), (const float*)(scores.buffer), (const float*)(classes.buffer), num_bboxes, detectionParams,
            networkInfo.width, networkInfo.height, (const float*)(landmarks.buffer));

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}


extern "C" bool NvDsInferParseCustomYolor(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    // {
    //     std::cerr << "WARNING: Num classes mismatch. Configured:"
    //               << detectionParams.numClassesConfigured
    //               << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    // }

    std::vector<NvDsInferParseObjectInfo> objects;
    const NvDsInferLayerInfo &n_bboxes   = outputLayersInfo[0];
    const NvDsInferLayerInfo &boxes      = outputLayersInfo[1]; // (num_boxes, 4)
    const NvDsInferLayerInfo &scores     = outputLayersInfo[2]; // (num_boxes, )
    const NvDsInferLayerInfo &classes    = outputLayersInfo[3]; // (num_boxes, )
    

    int num_bboxes = *(const int*)(n_bboxes.buffer);


    assert(boxes.inferDims.numDims == 2);
    assert(scores.inferDims.numDims == 1);
    assert(classes.inferDims.numDims == 1);

    // std::cout << "Network Info: " << networkInfo.height << "  " << networkInfo.width << std::endl;

    std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeYolorTensor(
            (const float*)(boxes.buffer), (const float*)(scores.buffer), (const float*)(classes.buffer), num_bboxes, detectionParams,
            networkInfo.width, networkInfo.height);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}

// extern "C" bool NvDsInferParseCustomSCRFD(
//     std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
//     NvDsInferNetworkInfo const& networkInfo,
//     NvDsInferParseDetectionParams const& detectionParams,
//     std::vector<NvDsInferParseObjectInfo>& objectList)
// {


//     std::vector<NvDsInferParseObjectInfo> objects;
//     const NvDsInferLayerInfo &n_bboxes   = outputLayersInfo[0];
//     const NvDsInferLayerInfo &boxes      = outputLayersInfo[1]; // (num_boxes, 4)
//     const NvDsInferLayerInfo &scores     = outputLayersInfo[2]; // (num_boxes, )
//     const NvDsInferLayerInfo &classes    = outputLayersInfo[3]; // (num_boxes, )
    

//     int num_bboxes = *(const int*)(n_bboxes.buffer);


//     assert(boxes.inferDims.numDims == 2);
//     assert(scores.inferDims.numDims == 1);
//     assert(classes.inferDims.numDims == 1);

//     std::vector<NvDsInferParseObjectInfo> outObjs =
//         decodeSCRFDTensor(
//             (const float*)(boxes.buffer), (const float*)(scores.buffer), (const float*)(classes.buffer), num_bboxes, detectionParams,
//             networkInfo.width, networkInfo.height);

//     objects.insert(objects.end(), outObjs.begin(), outObjs.end());

//     objectList = objects;

//     return true;
// }

/* YOLOv4 TLT with Padding*/
extern "C" bool NvDsInferParseCustomYoloV4TLT(
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList) {

     if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* Host memory for "BatchedNMS"
       BatchedNMS has 4 output bindings, the order is:
       keepCount, bboxes, scores, classes
    */
    int* p_keep_count = (int *) outputLayersInfo[0].buffer;
    float* p_bboxes = (float *) outputLayersInfo[1].buffer;
    float* p_scores = (float *) outputLayersInfo[2].buffer;
    float* p_classes = (float *) outputLayersInfo[3].buffer;

    const float threshold = detectionParams.perClassThreshold[0];
    const float pad_ratio_width = 0.01;
    const float pad_ratio_height = 0.04;

    const int keep_top_k = 200;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    // if(log_enable != NULL && std::stoi(log_enable)) {
    //     std::cout <<"keep cout"
    //           <<p_keep_count[0] << std::endl;
    // }

    for (int i = 0; i < p_keep_count[0] && objectList.size() <= keep_top_k; i++) {

        if ( p_scores[i] < threshold) continue;

        if(log_enable != NULL && std::stoi(log_enable)) {
            std::cout << "TLT label/conf/ x/y x/y -- "
                      << p_classes[i] << " " << p_scores[i] << " "
                      << p_bboxes[4*i] << " " << p_bboxes[4*i+1] << " " << p_bboxes[4*i+2] << " "<< p_bboxes[4*i+3] << " " << std::endl;
        }

        if((unsigned int) p_classes[i] >= detectionParams.numClassesConfigured) continue;
        if(p_bboxes[4*i+2] < p_bboxes[4*i] || p_bboxes[4*i+3] < p_bboxes[4*i+1]) continue;

        NvDsInferObjectDetectionInfo object;
        object.classId = (int) p_classes[i];
        object.detectionConfidence = p_scores[i];
        /* Clip object box co-ordinates to network resolution */
        float obj_left = CLIP(p_bboxes[4*i] * networkInfo.width, 0, networkInfo.width - 1);
        float obj_top = CLIP(p_bboxes[4*i+1] * networkInfo.height, 0, networkInfo.height - 1);
        float obj_width = CLIP(p_bboxes[4*i+2] * networkInfo.width, 0, networkInfo.width - 1) - obj_left;
        float obj_height = CLIP(p_bboxes[4*i+3] * networkInfo.height, 0, networkInfo.height - 1) - obj_top;
        
        /* Add padding*/
        float pad_width = pad_ratio_width * obj_width;
        float pad_height = pad_ratio_height * obj_height;
        object.left = CLIP(p_bboxes[4*i] * networkInfo.width - pad_width, 0, networkInfo.width - 1);
        object.top = CLIP(p_bboxes[4*i+1] * networkInfo.height - pad_height, 0, networkInfo.height - 1);
        object.width = CLIP(p_bboxes[4*i+2] * networkInfo.width + 2.0*pad_width, 0, networkInfo.width - 1) - object.left;
        object.height = CLIP(p_bboxes[4*i+3] * networkInfo.height + 2.0*pad_height, 0, networkInfo.height - 1) - object.top;

        if(object.height < 0 || object.width < 0)
            continue;
        objectList.push_back(object);
    }
    return true;
}


/* YOLOv4 TLT from detection to recognition*/
extern "C" bool NvDsInferParseCustomYoloV4LPR(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString) {

     if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* Host memory for "BatchedNMS"
       BatchedNMS has 4 output bindings, the order is:
       keepCount, bboxes, scores, classes
    */
    int* p_keep_count = (int *) outputLayersInfo[0].buffer;
    float* p_bboxes = (float *) outputLayersInfo[1].buffer;
    float* p_scores = (float *) outputLayersInfo[2].buffer;
    float* p_classes = (float *) outputLayersInfo[3].buffer;

    const int max_length = 10;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    // if(log_enable != NULL && std::stoi(log_enable)) {
    //     std::cout <<"keep cout"
    //           <<p_keep_count[0] << std::endl;
    // }
    // Read dict
    std::ifstream fdict;
    setlocale(LC_CTYPE, "");
    if(!DICT_LPR_READY) {
        fdict.open("dict.txt");
        if(!fdict.is_open())
        {
            std::cout << "open dictionary file failed." << std::endl;
            return false;
        }
        while(!fdict.eof()) {
            std::string strLineAnsi;
            if (getline(fdict, strLineAnsi) ) {
                if (strLineAnsi.length() > 1) {
                    strLineAnsi.erase(1);
                }
                DICT_LPR.push_back(strLineAnsi);
            }
        }
        DICT_LPR_READY=true;
        fdict.close();
    }

    // Empty list of object
    std::vector <ObjectPoint> objectList;
    // Append detection result
    for (int i = 0; i < p_keep_count[0] && objectList.size() <= max_length; i++) {

        if ( (float)p_scores[i] < classifierThreshold) continue;

        if(log_enable != NULL && std::stoi(log_enable)) {
            std::cout << "label/conf/ x/y x/y -- "
                  << p_classes[i] << " " << p_scores[i] << " "
                  << p_bboxes[4*i] << " " << p_bboxes[4*i+1] << " " << p_bboxes[4*i+2] << " "<< p_bboxes[4*i+3] << " " << std::endl;
        }

        if(p_bboxes[4*i+2] < p_bboxes[4*i] || p_bboxes[4*i+3] < p_bboxes[4*i+1]) continue;

        ObjectPoint obj;
        obj.ctx = (float)(p_bboxes[4*i] + p_bboxes[4*i+2])/2.0;
        obj.cty = (float)(p_bboxes[4*i+1] + p_bboxes[4*i+3])/2.0;
        obj.width = (float)p_bboxes[4*i+2] - p_bboxes[4*i];
        obj.height = (float)p_bboxes[4*i+3] - p_bboxes[4*i+1];
        obj.confidence = (float)p_scores[i];
        obj.classId = (int) p_classes[i];

        if(obj.height < 0 || obj.width < 0)
            continue;
        objectList.push_back(obj);
    }
    // Add to metadata
    NvDsInferAttribute LPR_attr;
    // LPR_attr.attributeConfidence = sumConfidence / objectListConfidence.size();
    attrString = mergeDetectionResult(objectList);
    if (objectList.size() >=  3) {
        LPR_attr.attributeIndex = 0;
        LPR_attr.attributeValue = 1;
        LPR_attr.attributeConfidence = 1.0;
        LPR_attr.attributeLabel = strdup(attrString.c_str());
        for (ObjectPoint obj: objectList)
            LPR_attr.attributeConfidence *= obj.confidence;
        attrList.push_back(LPR_attr);
        if(log_enable != NULL && std::stoi(log_enable)) {
            std::cout << "License plate: " << attrString << "  -  Confidence: " << LPR_attr.attributeConfidence << std::endl;
        }
    }

    return true;
}

/* Object color*/
extern "C" bool NvDsInferParseCustomObjColors(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString) {

     if(outputLayersInfo.size() != 1)
    {
        std::cerr << "Expected 1 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    int32_t* obj_colors = (int32_t *) outputLayersInfo[0].buffer;

    const int topK = 3;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    // Object color to string
    int cc = 0;
    // std::string obj_color_str = "";
    for(int i = 0; i < topK; ++i){
        if (i != 0) attrString += ",";
        attrString += "(";
        attrString += std::to_string(obj_colors[cc]);
        attrString += ",";
        attrString += std::to_string(obj_colors[cc + 1]);
        attrString += ",";
        attrString += std::to_string(obj_colors[cc + 2]);
        attrString += ",";
        attrString += std::to_string(obj_colors[cc + 3]);
        attrString += ")";
        cc += 4;
    }
    // LPR_attr.attributeConfidence = sumConfidence / objectListConfidence.size();
    // Add to metadata
    NvDsInferAttribute obj_color_attr;

    obj_color_attr.attributeIndex = 0;
    obj_color_attr.attributeValue = 1;
    obj_color_attr.attributeConfidence = 1.0;
    obj_color_attr.attributeLabel = strdup(attrString.c_str());

    attrList.push_back(obj_color_attr);
    if(log_enable != NULL && std::stoi(log_enable)) {
        std::cout << "Color: " << obj_color_attr.attributeLabel  << std::endl;

    }

    return true;
}

extern "C" bool NvDsInferParseCustomPersonColors(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString) {

    if(outputLayersInfo.size() != 1)
    {
        std::cerr << "Expected 1 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    int32_t* human_colors      = (int32_t *) outputLayersInfo[0].buffer;
    

    const int topK = 3;
    const int nPart = 4; 
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    // Object color to string, divided into 4 parts
    std::vector <std::string> colorParts;
    int cc = 0;
    for (int k = 0; k < nPart; k++){
        std::string colorPart = "";
        for (int i = 0; i < topK; i++){
            if (i != 0) colorPart += ',';
            colorPart += '(';
            colorPart += std::to_string(human_colors[cc]);
            colorPart += ",";
            colorPart += std::to_string(human_colors[cc + 1]);
            colorPart += ",";
            colorPart += std::to_string(human_colors[cc + 2]);
            colorPart += ',';
            colorPart += std::to_string(human_colors[cc + 3]);
            colorPart += ')';
            cc += 4;
        };
        colorParts.push_back(colorPart);
    }
    attrString = colorParts[0];
    
    // LPR_attr.attributeConfidence = sumConfidence / objectListConfidence.size();
    // Add to metadata
    // attrString += "abcd";
    // descString += person_color_str;
    // std::string tmp = "test";
    for (int k = 0; k < nPart; k++){
        NvDsInferAttribute person_color_attr;
        person_color_attr.attributeIndex = k;
        person_color_attr.attributeValue = 2;
        person_color_attr.attributeConfidence = 1.0;
        person_color_attr.attributeLabel = strdup(colorParts[k].c_str());
        attrList.push_back(person_color_attr);
    }
    if(log_enable != NULL && std::stoi(log_enable)) {
        for (int k = 0; k < nPart; k++){
            std::cout << "Human Color Part: " << attrList[k].attributeIndex << " String: " << attrList[k].attributeLabel << std::endl;
        }
    }

    return true;
}

extern "C" bool NvDsInferParseCustomPersonColorsV2(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString) {

     if(outputLayersInfo.size() != 2)
    {
        std::cerr << "Expected 2 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    int32_t* human_colors = (int32_t *) outputLayersInfo[1].buffer;
    float* human_colors_conf   = (float *) outputLayersInfo[0].buffer;

    const int topK = 3;
    const int nPart = 8; // upperbody_color, lowerbody_color, hat_color, motorbike_color, shoe_color, handpack_color, watch_color, phone_color
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    // Object color to string, divided into 4 parts
    std::vector <std::string> colorParts;
    int cc = 0;
    for (int k = 0; k < nPart; k++){
        std::string colorPart = "";
        for (int i = 0; i < topK; i++){
            if (i != 0) colorPart += ',';
            colorPart += '(';
            colorPart += std::to_string(human_colors[cc]);
            colorPart += ",";
            colorPart += std::to_string(human_colors[cc + 1]);
            colorPart += ",";
            colorPart += std::to_string(human_colors[cc + 2]);
            colorPart += ',';
            colorPart += std::to_string(human_colors[cc + 3]);
            colorPart += ')';
            cc += 4;
        };
        colorParts.push_back(colorPart);
    }
    attrString = colorParts[0];
    
    // LPR_attr.attributeConfidence = sumConfidence / objectListConfidence.size();
    // Add to metadata
    // attrString += "abcd";
    // descString += person_color_str;
    // std::string tmp = "test";
    for (int k = 0; k < nPart; k++){
        NvDsInferAttribute person_color_attr;
        person_color_attr.attributeIndex = k;
        person_color_attr.attributeValue = 2;
        person_color_attr.attributeConfidence = human_colors_conf[k];
        person_color_attr.attributeLabel = strdup(colorParts[k].c_str());
        attrList.push_back(person_color_attr);
    }
    if(log_enable != NULL && std::stoi(log_enable)) {
        for (int k = 0; k < nPart; k++){
            std::cout << "Human Color Part: " << attrList[k].attributeIndex << " String: " << attrList[k].attributeLabel << std::endl;
        }
    }

    return true;
}

extern "C" bool NvDsInferParseCustomCarColors(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString) {

     if(outputLayersInfo.size() != 1)
    {
        std::cerr << "Expected 1 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    int32_t* car_colors = (int32_t *) outputLayersInfo[0].buffer;

    const int topK = 3;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    // Object color to string
    int cc = 0;
    for (int i = 0; i < topK; i++){
        if (i != 0) attrString += ',';
        attrString += '(';
        attrString += std::to_string(car_colors[cc]);
        attrString += ",";
        attrString += std::to_string(car_colors[cc + 1]);
        attrString += ",";
        attrString += std::to_string(car_colors[cc + 2]);
        attrString += ',';
        attrString += std::to_string(car_colors[cc + 3]);
        attrString += ')';
        cc += 4;
    };

    
    // LPR_attr.attributeConfidence = sumConfidence / objectListConfidence.size();
    // Add to metadata
    // attrString += "abcd";
    // descString += person_color_str;
    // std::string tmp = "test";

    NvDsInferAttribute car_color_attr;
    car_color_attr.attributeIndex = 0;
    car_color_attr.attributeValue = 1;
    car_color_attr.attributeConfidence = 1.0;
    car_color_attr.attributeLabel = strdup(attrString.c_str());
    attrList.push_back(car_color_attr);

    if(log_enable != NULL && std::stoi(log_enable)) {
        std::cout << "Car Color Part: " << attrList[0].attributeIndex << " String: " << attrList[0].attributeLabel << std::endl;

    }

    return true;
}

extern "C" bool NvDsInferParseCustomCarColorsV2(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString) {

     if(outputLayersInfo.size() != 3)
    {
        std::cerr << "Expected 3 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }
    int32_t* car_palette_colors = (int32_t *) outputLayersInfo[0].buffer;
    float_t* car_color_conf     = (float_t *) outputLayersInfo[1].buffer;
    int32_t* car_color          = (int32_t *) outputLayersInfo[2].buffer;

    const int topK = 3;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    // Object color to string
    int cc = 0;
    for (int i = 0; i < topK; i++){
        if (i != 0) attrString += ',';
        attrString += '(';
        attrString += std::to_string(car_palette_colors[cc]);
        attrString += ",";
        attrString += std::to_string(car_palette_colors[cc + 1]);
        attrString += ",";
        attrString += std::to_string(car_palette_colors[cc + 2]);
        attrString += ',';
        attrString += std::to_string(car_palette_colors[cc + 3]);
        attrString += ')';
        cc += 4;
    };
    // Domination color
    for (int i = 0; i < 4; i++){
        if (i == 0){
            attrString += ";";
        }
        else{
            attrString += ",";
        }
        attrString += std::to_string(car_color[i]);
    }
    // std::cout << "Palette: " << attrString << std::endl;
    
    
    // LPR_attr.attributeConfidence = sumConfidence / objectListConfidence.size();
    // Add to metadata
    // attrString += "abcd";
    // descString += person_color_str;
    // std::string tmp = "test";

    NvDsInferAttribute car_color_attr;
    car_color_attr.attributeIndex = 0;
    car_color_attr.attributeValue = 1;
    car_color_attr.attributeConfidence = car_color_conf[0];
    car_color_attr.attributeLabel = strdup(attrString.c_str());
    attrList.push_back(car_color_attr);

    if(log_enable != NULL && std::stoi(log_enable)) {
        std::cout << "Car Color Part: " << attrList[0].attributeIndex << " String: " << attrList[0].attributeLabel << std::endl;

    }

    return true;
}

extern "C" bool NvDsInferParseCustomFaceEmbedding(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString)
{



    int* num_detections       = (int*) outputLayersInfo[5].buffer;
    float* boxes              = (float*) outputLayersInfo[4].buffer; // (num_boxes, 4)
    float* scores             = (float*) outputLayersInfo[3].buffer; // (num_boxes, )
    float* face_align         = (float*) outputLayersInfo[2].buffer; // (num_boxes, 3, 112, 112)
    float* landmarks          = (float*) outputLayersInfo[1].buffer; // (num_boxes, 5, 2)
    float* embeddings         = (float*) outputLayersInfo[0].buffer; // (num_boxes, 512)

    const int number_features = 512;
    const int n_truncate = 43; // 43 truncates, each truncate have 12 value: 12, 12, 12, ... 12, 8
    const int truncate_size = 12;
    const char* log_enable = std::getenv("ENABLE_DEBUG");
    // for (int i = 0; i < 6; i++){
    //     std::cout <<  outputLayersInfo[i].layerName << std::endl;
    // }
    int num_bboxes = *(const int*)(num_detections);
    // std::cout << "number of boxes: " << num_bboxes << std::endl;
    // std::cout << "boxes: " << boxes.layerName << std::endl;
    // std::cout << "scores: " << scores.layerName << std::endl;
    // std::cout << "landmarks: " << landmarks.layerName << std::endl;
    // assert(num_detections.inferDims.numDims == 1);
    // assert(boxes.inferDims.numDims == 2);
    // assert(scores.inferDims.numDims == 1);
    // assert(face_align.inferDims.numDims == 4);
    // assert(landmarks.inferDims.numDims == 3);
    // assert(embeddings.inferDims.numDims == 2);

    if (num_bboxes > 0){
        assert(num_bboxes == 1); // Fixed value
        // Box string
        std::string bbox_str = std::to_string(boxes[0]) + "," + std::to_string(boxes[1]) + "," + std::to_string(boxes[2]) + "," + std::to_string(boxes[3]);
        std::string lmk_str  = "";
        for (int i = 0; i < 10; i++){
            if (i != 0) lmk_str += ",";
            lmk_str += std::to_string(landmarks[i]);
        }
        lmk_str += ",";
        lmk_str += std::to_string(scores[0]);

        attrString += bbox_str;
        // Add to metadata
        int idx = 0;
        for (int k = 0; k < n_truncate; k++){
            std::string truncate_label = "";
            for (int kk = 0; kk < truncate_size; kk++){
                if (kk != 0) truncate_label += ",";
                truncate_label += std::to_string(embeddings[idx]);
                idx += 1;
                if (idx == number_features) {
                    // Concat with 4 box co-ordinate 
                    truncate_label += ",";
                    truncate_label += bbox_str;
                    break;
                }
            }
            NvDsInferAttribute face_embed_truncate;
            face_embed_truncate.attributeIndex = k;
            face_embed_truncate.attributeValue = 1;
            face_embed_truncate.attributeConfidence = 1.0;
            face_embed_truncate.attributeLabel = strdup(truncate_label.c_str());
            // std::cout << "push back " << k << std::endl;
            attrList.push_back(face_embed_truncate);
        }

        // Additional landmark
        NvDsInferAttribute face_embed_truncate;
        face_embed_truncate.attributeIndex = n_truncate;
        face_embed_truncate.attributeValue = 1;
        face_embed_truncate.attributeConfidence = scores[0];
        face_embed_truncate.attributeLabel = strdup(lmk_str.c_str());
        // std::cout << "push back " << k << std::endl;
        attrList.push_back(face_embed_truncate);
    }


    return true;
}




extern "C" bool NvDsInferParseCustomFaceEmbeddingAttribute(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString)
{


    // for (int i = 0; i < 8; i++){
    //     std::cout <<  outputLayersInfo[i].layerName << std::endl;
    // }
    int* res_num_detections       = (int*)   outputLayersInfo[7].buffer;
    float* res_bboxes             = (float*) outputLayersInfo[5].buffer; // (num_boxes, 4)
    float* res_scores             = (float*) outputLayersInfo[4].buffer; // (num_boxes, )
    float* res_landmarks          = (float*) outputLayersInfo[3].buffer; // (num_boxes, 5, 2)
    float* res_embedding          = (float*) outputLayersInfo[2].buffer; // (num_boxes, 512)
    float* res_gender             = (float*) outputLayersInfo[6].buffer; // (num_boxes, 512)
    float* res_glass              = (float*) outputLayersInfo[1].buffer; // (num_boxes, 512)
    float* res_mask               = (float*) outputLayersInfo[0].buffer; // (num_boxes, 512)

    const int number_features = 512;
    const int n_truncate = 43; // 43 truncates, each truncate have 12 value: 12, 12, 12, ... 12, 8
    const int truncate_size = 12;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    int num_bboxes = *(const int*)(res_num_detections);


    if (num_bboxes > 0){
        // std::cout << "Face found " << std::endl;
        assert(num_bboxes == 1); // Fixed value
        // Box string
        std::string bbox_str = std::to_string(res_bboxes[0]) + "," + std::to_string(res_bboxes[1]) + "," + std::to_string(res_bboxes[2]) + "," + std::to_string(res_bboxes[3]);
        // std::cout << bbox_str << std::endl;
        std::string lmk_str  = "";
        for (int i = 0; i < 10; i++){
            if (i != 0) lmk_str += ",";
            lmk_str += std::to_string(res_landmarks[i]);
        }
        lmk_str += ",";
        lmk_str += std::to_string(res_scores[0]);

        attrString += bbox_str;
        // Add to metadata
        int idx = 0;
        for (int k = 0; k < n_truncate; k++){
            std::string truncate_label = "";
            for (int kk = 0; kk < truncate_size; kk++){
                if (kk != 0) truncate_label += ",";
                truncate_label += std::to_string(res_embedding[idx]);
                idx += 1;
                if (idx == number_features) {
                    // Concat with 4 box co-ordinate 
                    truncate_label += ",";
                    truncate_label += bbox_str;
                    break;
                }
            }
            NvDsInferAttribute face_embed_truncate;
            face_embed_truncate.attributeIndex = k;
            face_embed_truncate.attributeValue = 1;
            face_embed_truncate.attributeConfidence = 1.0;
            face_embed_truncate.attributeLabel = strdup(truncate_label.c_str());
            // std::cout << "push back " << k << std::endl;
            attrList.push_back(face_embed_truncate);
        }

        // Additional landmark
        NvDsInferAttribute face_embed_truncate;
        face_embed_truncate.attributeIndex = n_truncate;
        face_embed_truncate.attributeValue = 1;
        face_embed_truncate.attributeConfidence = res_scores[0];
        face_embed_truncate.attributeLabel = strdup(lmk_str.c_str());
        attrList.push_back(face_embed_truncate);
        // Additional attribute
        // Gender
        std::string gender_str = std::to_string(res_gender[1]);
        NvDsInferAttribute face_gender_truncate;
        face_gender_truncate.attributeIndex = n_truncate + 1;
        face_gender_truncate.attributeValue = 1;
        face_gender_truncate.attributeConfidence = res_gender[1];
        face_gender_truncate.attributeLabel = strdup(gender_str.c_str());
        attrList.push_back(face_gender_truncate);
        // Glass
        std::string glass_str = std::to_string(res_glass[1]);
        NvDsInferAttribute face_glass_truncate;
        face_glass_truncate.attributeIndex = n_truncate + 2;
        face_glass_truncate.attributeValue = 1;
        face_glass_truncate.attributeConfidence = res_glass[1];
        face_glass_truncate.attributeLabel = strdup(glass_str.c_str());
        attrList.push_back(face_glass_truncate);
        // Mask
        std::string mask_str = std::to_string(res_mask[1]);
        NvDsInferAttribute face_mask_truncate;
        face_mask_truncate.attributeIndex = n_truncate + 3;
        face_mask_truncate.attributeValue = 1;
        face_mask_truncate.attributeConfidence = res_mask[1];
        face_mask_truncate.attributeLabel = strdup(mask_str.c_str());
        attrList.push_back(face_mask_truncate);
    }
    else{
        // std::cout << "Face not found " << std::endl;
        std::string not_found = "not found";
        attrString += not_found;
        // NvDsInferAttribute face_not_found;
        // face_not_found.attributeIndex = 0;
        // face_not_found.attributeValue = 0;
        // face_not_found.attributeConfidence = 1.0;
        // face_not_found.attributeLabel = strdup(not_found.c_str());
        // attrList.push_back(face_not_found);
    }

    return true;
}

extern "C" bool NvDsInferParseCustomYoloV4VMN(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString) {

    if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* Host memory for "BatchedNMS"
       BatchedNMS has 4 output bindings, the order is:
       keepCount, bboxes, scores, classes
    */
    int* p_keep_count = (int *) outputLayersInfo[0].buffer;
    float* p_bboxes = (float *) outputLayersInfo[1].buffer;
    float* p_scores = (float *) outputLayersInfo[2].buffer;
    float* p_classes = (float *) outputLayersInfo[3].buffer;
    const char* log_enable = std::getenv("ENABLE_DEBUG_VMN");
    std::ifstream fdict;
    setlocale(LC_CTYPE, "");
    if(!DICT_VMN_READY) {
        fdict.open("weights/vehiclemakenet/labels.txt");
        if(!fdict.is_open())
        {
            std::cout << "open dictionary file failed." << std::endl;
            return false;
        }
        while(!fdict.eof()) {
            std::string strLineAnsi;
            if (getline(fdict, strLineAnsi) ) {
                // std::cout << strLineAnsi << std::endl;
                // if (strLineAnsi.length() > 1) {
                //     strLineAnsi.erase(1);
                strLineAnsi.erase(strLineAnsi.size() - 1);
                // }
                // std::cout << strLineAnsi << std::endl;
                DICT_VMN.push_back(strLineAnsi);
            }
        }
        DICT_VMN_READY=true;
        fdict.close();
    }


    if ( p_scores[0] > classifierThreshold){


        NvDsInferAttribute VMN_attr;
        attrString = DICT_VMN[(int) p_classes[0]];

        VMN_attr.attributeIndex = 0;
        VMN_attr.attributeValue = 1;
        VMN_attr.attributeConfidence = p_scores[0];
        VMN_attr.attributeLabel = strdup(attrString.c_str());

        attrList.push_back(VMN_attr);
        if(log_enable != NULL && std::stoi(log_enable)) {
            std::cout << "Vehicle Make: " << (int) p_classes[0] << " " << attrString << "  -  Confidence: " << VMN_attr.attributeConfidence << std::endl;
        }
    }


    return true;
}


/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomRLP);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolor);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4TLT);
// CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSCRFD);
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4LPR);
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomObjColors);
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomPersonColors);
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomPersonColorsV2);
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomCarColors);
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomCarColorsV2);
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomFaceEmbedding);
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4VMN);