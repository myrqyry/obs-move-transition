#include "mediapipe-load.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

template<typename T> T sym(void* handle, const char* name) {
 void *p=dlsym(handle,name);T typed{};memcpy(&typed,&p,sizeof(p));return typed;
}
int main(int argc,char**argv){
 if(argc!=5)return 2;
 void *lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
 if(!lib){std::fprintf(stderr,"DLOPEN %s\n",dlerror());return 3;}
 auto create_image=sym<decltype(mp_api_t::image_create_from_uint8_data)>(lib,"MpImageCreateFromUint8Data");
 auto free_image=sym<decltype(mp_api_t::image_free)>(lib,"MpImageFree");
 auto free_error=sym<decltype(mp_api_t::error_free)>(lib,"MpErrorFree");
 auto face_create=sym<decltype(mp_api_t::face_landmarker_create)>(lib,"MpFaceLandmarkerCreate");
 auto face_detect=sym<decltype(mp_api_t::face_landmarker_detect_image)>(lib,"MpFaceLandmarkerDetectImage");
 auto face_close=sym<decltype(mp_api_t::face_landmarker_close)>(lib,"MpFaceLandmarkerClose");
 auto face_result_close=sym<decltype(mp_api_t::face_landmarker_close_result)>(lib,"MpFaceLandmarkerCloseResult");
 auto pose_create=sym<decltype(mp_api_t::pose_landmarker_create)>(lib,"MpPoseLandmarkerCreate");
 auto pose_detect=sym<decltype(mp_api_t::pose_landmarker_detect_image)>(lib,"MpPoseLandmarkerDetectImage");
 auto pose_close=sym<decltype(mp_api_t::pose_landmarker_close)>(lib,"MpPoseLandmarkerClose");
 auto pose_result_close=sym<decltype(mp_api_t::pose_landmarker_close_result)>(lib,"MpPoseLandmarkerCloseResult");
 std::ifstream input(argv[4],std::ios::binary);
 std::vector<unsigned char> raw(256*256*4);
 if(!input.read(reinterpret_cast<char*>(raw.data()),raw.size()))return 4;
 MpImage *img=nullptr;char *err=nullptr;
 int ret=create_image(2,256,256,raw.data(),(int)raw.size(),&img,&err);
 printf("TASKS_IMAGE status=%d valid=%d err=%s\n",ret,img!=nullptr,err?err:"none");
 if(err){free_error(err);err=nullptr;} if(ret||!img)return 5;
 FaceLandmarkerOptionsC face{};
 face.base_options.model_asset_path=argv[2];face.running_mode=1;face.num_faces=1;
 face.min_face_detection_confidence=0.5f;face.min_face_presence_confidence=0.5f;
 face.min_tracking_confidence=0.5f;
 face.output_face_blendshapes=true;face.output_facial_transformation_matrixes=true;
 void *fh=nullptr;
 ret=face_create(&face,&fh,&err);
 printf("TASKS_FACE_CREATE status=%d handle=%d err=%s\n",ret,fh!=nullptr,err?err:"none");
 if(err){free_error(err);err=nullptr;}
 bool faceOk=false;
 if(!ret&&fh){
  FaceLandmarkerResultC out{};
  ret=face_detect(fh,img,nullptr,&out,&err);
  float loX=2,hiX=-1,loY=2,hiY=-1;
  unsigned n=out.face_landmarks_count ? out.face_landmarks[0].landmarks_count:0;
  if(n>0&&n<1000&&out.face_landmarks[0].landmarks){
   for(unsigned i=0;i<n;i++){auto &p=out.face_landmarks[0].landmarks[i];loX=std::min(loX,p.x);hiX=std::max(hiX,p.x);loY=std::min(loY,p.y);hiY=std::max(hiY,p.y);}
  }
  printf("TASKS_FACE_RESULT status=%d faces=%u landmarks=%u x=[%.3f,%.3f] y=[%.3f,%.3f] blendshapes=%u matrices=%u err=%s\n",ret,out.face_landmarks_count,n,loX,hiX,loY,hiY,out.face_blendshapes_count,out.facial_transformation_matrixes_count,err?err:"none");
  faceOk=ret==0&&n>=468&&n<=478&&hiX-loX>0.2f&&hiY-loY>0.2f;
  if(err){free_error(err);err=nullptr;}
  if(!ret)face_result_close(&out);
  face_close(fh,&err);if(err){free_error(err);err=nullptr;}
 }
 PoseLandmarkerOptionsC pose{};
 pose.base_options.model_asset_path=argv[3];pose.running_mode=1;pose.num_poses=1;
 pose.min_pose_detection_confidence=0.5f;pose.min_pose_presence_confidence=0.5f;
 pose.min_tracking_confidence=0.5f;
 void *ph=nullptr;
 ret=pose_create(&pose,&ph,&err);
 printf("TASKS_POSE_CREATE status=%d handle=%d err=%s\n",ret,ph!=nullptr,err?err:"none");
 if(err){free_error(err);err=nullptr;}
 bool poseOk=false;
 if(ret==0&&ph){
  PoseLandmarkerResultC out{};
  ret=pose_detect(ph,img,nullptr,&out,&err);
  unsigned n=out.pose_landmarks_count && out.pose_landmarks ? out.pose_landmarks[0].landmarks_count :0;
  printf("TASKS_POSE_RESULT status=%d poses=%u landmarks=%u err=%s\n",ret,out.pose_landmarks_count,n,err?err:"none");
  poseOk=ret==0&&n>0&&n<=40;
  if(err){free_error(err);err=nullptr;}
  if(!ret)pose_result_close(&out);
  pose_close(ph,&err);if(err){free_error(err);err=nullptr;}
 }
 free_image(img);dlclose(lib);
 printf("TASKS_PASS face=%d pose=%d\n",faceOk,poseOk);
 return faceOk&&poseOk?0:1;
}
