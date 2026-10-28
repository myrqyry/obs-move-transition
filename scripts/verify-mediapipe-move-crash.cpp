#include <obs.h>
#include <obs-module.h>
#include <graphics/graphics.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
static std::vector<uint8_t> face_pixels;
static const char *name(void*){return "Crash repro video source";}
struct Fixture {gs_texture_t *texture = nullptr;};
static Fixture *active_fixture = nullptr;
static void* create(obs_data_t*,obs_source_t*){
 auto *f=new Fixture;
 obs_enter_graphics();
 f->texture=gs_texture_create(256,256,GS_RGBA,1,nullptr,GS_DYNAMIC);
 if(f->texture && face_pixels.size()==256*256*4)
   gs_texture_set_image(f->texture,face_pixels.data(),256*4,false);
 obs_leave_graphics();
 active_fixture=f;return f;
}
static void destroy(void*p){
 auto*f=(Fixture*)p;obs_enter_graphics();
 if(f->texture)gs_texture_destroy(f->texture);
 obs_leave_graphics();active_fixture=nullptr;delete f;
}
static uint32_t width(void*){return 256;}
static uint32_t height(void*){return 256;}
static void render(void*p,gs_effect_t*) {
 auto*f=(Fixture*)p;
 if(f->texture)obs_source_draw(f->texture,0,0,256,256,false);
}
static obs_source_info input={
 .id="mp_move_crash_fixture", .type=OBS_SOURCE_TYPE_INPUT,
 .output_flags=OBS_SOURCE_VIDEO,.get_name=name,.create=create,.destroy=destroy,
 .get_width=width,.get_height=height,.video_render=render};
int main(int argc,char**argv){
 if(argc<4)return 2;
 setbuf(stdout,nullptr);
 puts("REPRO_START");
 if(argc>5){
   std::ifstream file(argv[5],std::ios::binary);
   face_pixels.resize(256*256*4);
   if(!file.read(reinterpret_cast<char*>(face_pixels.data()),face_pixels.size()))return 9;
 }
 if(!obs_startup("en-US",nullptr,nullptr))return 3;
 puts("REPRO_OBS_STARTUP");
 obs_video_info v{};v.graphics_module="libobs-opengl";v.fps_num=30;v.fps_den=1;
 v.base_width=256;v.base_height=256;v.output_width=256;v.output_height=256;
 v.output_format=VIDEO_FORMAT_RGBA;v.scale_type=OBS_SCALE_DISABLE;
 if(obs_reset_video(&v)!=OBS_VIDEO_SUCCESS)return 4;
 puts("REPRO_OBS_VIDEO");
 obs_register_source(&input);
 obs_module_t*mod=nullptr;
 auto status=obs_open_module(&mod,argv[1],argv[2]);
 printf("REPRO_MODULE_OPEN status=%d\n",(int)status);
 if(status!=MODULE_SUCCESS || !obs_init_module(mod))return 5;
 puts("REPRO_MODULE_READY");
 auto*src=obs_source_create("mp_move_crash_fixture","fixture",nullptr,nullptr);
 puts("REPRO_INPUT_CREATED");
 obs_data_t*settings=obs_data_create();
 if(!strcmp(argv[3],"face")){
   obs_data_set_int(settings,"actions",1);
   obs_data_set_int(settings,"action_1_feature",1);
 } else if(!strcmp(argv[3],"body")) {
   obs_data_set_int(settings,"actions",1);
   obs_data_set_int(settings,"action_1_feature",5);
 } else if(!strcmp(argv[3],"both")) {
   obs_data_set_int(settings,"actions",2);
   obs_data_set_int(settings,"action_1_feature",1);
   obs_data_set_int(settings,"action_2_feature",5);
 }
 puts("REPRO_CREATE_FILTER");
 auto*filter=obs_source_create("mp_move_filter","test MediaPipe Move",settings,nullptr);
 printf("REPRO_FILTER_CREATED=%p\n",(void*)filter);
 obs_data_release(settings);
 if(!filter)return 6;
 puts("REPRO_ATTACH");
 obs_source_filter_add(src,filter);
 puts("REPRO_ATTACHED");
 if(argc>4 && !strcmp(argv[4],"render")){
   for(int i=0;i<24;i++){
    if(i==12 && face_pixels.size()==256*256*4 && active_fixture && active_fixture->texture){
      std::vector<uint8_t> blank(256*256*4,0);
      for(size_t px=3;px<blank.size();px+=4)blank[px]=255;
      obs_enter_graphics();
      gs_texture_set_image(active_fixture->texture,blank.data(),256*4,false);
      obs_leave_graphics();
      puts("REPRO_SWITCHED_TO_BLANK");
    }
    puts("REPRO_RENDER");
    obs_enter_graphics();
    auto *target=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    if(target && gs_texrender_begin(target,256,256)){
      gs_projection_push();gs_ortho(0,256,0,256,-100,100);
      obs_source_video_render(src);
      gs_projection_pop();
      gs_texrender_end(target);
    }
    if(target)gs_texrender_destroy(target);
    obs_leave_graphics();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
   }
 }
 puts("REPRO_DETACH");
 obs_source_filter_remove(src,filter);
 obs_source_release(filter);
 obs_source_release(src);
 puts("REPRO_SHUTDOWN");
 obs_shutdown();
 puts("REPRO_PASS");
 return 0;
}
