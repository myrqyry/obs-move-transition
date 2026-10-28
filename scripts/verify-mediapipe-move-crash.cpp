#include <obs.h>
#include <obs-module.h>
#include <graphics/graphics.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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
 const bool attach_mode=!strcmp(argv[3],"attach");
 const bool body_move_mode=!strcmp(argv[3],"body-move");
 const bool expression_mode=!strcmp(argv[3],"expression-vector");
 const bool pose_mode=!strcmp(argv[3],"pose-value");
 const bool move_mode=!strcmp(argv[3],"move") || attach_mode || body_move_mode || expression_mode;
 obs_scene_t *test_scene=nullptr;
 obs_sceneitem_t *test_item=nullptr;
 if(move_mode) {
   test_scene=obs_scene_create("test scene");
   auto *overlay=obs_source_create("mp_move_crash_fixture","overlay",nullptr,nullptr);
   test_item=obs_scene_add(test_scene,overlay);
   obs_source_release(overlay);
 }
 obs_data_t*settings=obs_data_create();
 if(!strcmp(argv[3],"face")){
   obs_data_set_int(settings,"actions",1);
   obs_data_set_int(settings,"action_1_feature",1);
 } else if(!strcmp(argv[3],"body")) {
   obs_data_set_int(settings,"actions",1);
   obs_data_set_int(settings,"action_1_feature",5);
 } else if(!strcmp(argv[3],"reorder")) {
   obs_data_set_int(settings,"actions",2);
   obs_data_set_string(settings,"action_1_description","first");
   obs_data_set_string(settings,"action_2_description","second");
   obs_data_set_int(settings,"action_1_feature",1);
   obs_data_set_int(settings,"action_2_feature",5);
   obs_data_set_double(settings,"action_1_factor",18.0);
   obs_data_set_double(settings,"action_2_factor",35.0);
   obs_data_set_string(settings,"action_2_canvas","Custom");
 } else if(!strcmp(argv[3],"missing-filter")) {
   obs_data_set_int(settings,"actions",1);
   obs_data_set_int(settings,"action_1_action",2);
   obs_data_set_string(settings,"action_1_source","fixture");
   obs_data_set_string(settings,"action_1_filter","nonexistent_filter");
 } else if(!strcmp(argv[3],"missing-value")) {
   obs_data_set_int(settings,"actions",1);
   obs_data_set_int(settings,"action_1_action",1);
   obs_data_set_string(settings,"action_1_source","fixture");
   obs_data_set_string(settings,"action_1_property","nonexistent_property");
 } else if(attach_mode) {
   obs_data_set_int(settings,"actions",1);
   obs_data_set_int(settings,"action_1_action",4); // attach source
   obs_data_set_string(settings,"action_1_scene","test scene");
   obs_data_set_string(settings,"action_1_sceneitem","overlay");
   obs_data_set_int(settings,"action_1_attach",9); // nose
   obs_data_set_double(settings,"action_1_factor",100);
   obs_data_set_double(settings,"action_1_factor2",100);
 } else if(pose_mode) {
   obs_data_set_int(settings,"actions",1);
   obs_data_set_int(settings,"action_1_feature",2);
   obs_data_set_int(settings,"action_1_pose",0); // quaternion X
 } else if(expression_mode) {
   obs_data_set_int(settings,"actions",1);
   obs_data_set_int(settings,"action_1_action",0);
   obs_data_set_string(settings,"action_1_scene","test scene");
   obs_data_set_string(settings,"action_1_sceneitem","overlay");
   obs_data_set_int(settings,"action_1_sceneitem_property",1); // vec2 position
   obs_data_set_int(settings,"action_1_feature",3); // expressions
   obs_data_set_int(settings,"action_1_expression_property",1); // vector
   obs_data_set_int(settings,"action_1_expression_1",5); // browOuterUpLeft
   obs_data_set_int(settings,"action_1_expression_2",14); // eyeLookInLeft
   obs_data_set_double(settings,"action_1_factor",10000);
   obs_data_set_double(settings,"action_1_factor2",10000);
 } else if(body_move_mode) {
   obs_data_set_int(settings,"actions",1);
   obs_data_set_int(settings,"action_1_action",0);
   obs_data_set_string(settings,"action_1_scene","test scene");
   obs_data_set_string(settings,"action_1_sceneitem","overlay");
   obs_data_set_int(settings,"action_1_sceneitem_property",2);
   obs_data_set_int(settings,"action_1_feature",5); // body
   obs_data_set_int(settings,"action_1_body",1); // 2D X
   obs_data_set_int(settings,"action_1_body_1",0); // nose
   obs_data_set_double(settings,"action_1_factor",100);
 } else if(move_mode) {
   obs_data_set_int(settings,"actions",2);
   obs_data_set_int(settings,"action_1_action",0); // move source
   obs_data_set_string(settings,"action_1_scene","test scene");
   obs_data_set_string(settings,"action_1_sceneitem","overlay");
   obs_data_set_int(settings,"action_1_sceneitem_property",2); // position X
   obs_data_set_int(settings,"action_1_feature",0); // bounding box
   obs_data_set_int(settings,"action_1_bounding_box",0); // face left
   obs_data_set_double(settings,"action_1_factor",100);
   obs_data_set_int(settings,"action_2_action",3); // visibility
   obs_data_set_string(settings,"action_2_scene","test scene");
   obs_data_set_string(settings,"action_2_sceneitem","overlay");
   obs_data_set_int(settings,"action_2_feature",0);
   obs_data_set_int(settings,"action_2_bounding_box",6); // width
   obs_data_set_int(settings,"action_2_enable",3); // disable over
   obs_data_set_double(settings,"action_2_threshold",0.1);
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
 if(!strcmp(argv[3],"reorder")){
   auto *props=obs_source_properties(filter);
   auto *group=obs_properties_get(props,"action_2_group");
   auto *sub=group ? obs_property_group_content(group):nullptr;
   auto *button=sub ? obs_properties_get(sub,"action_2_move_up"):nullptr;
   bool clicked=button && obs_property_button_clicked(button,filter);
   auto *updated=obs_source_get_settings(filter);
   bool swapped=clicked &&
       !strcmp(obs_data_get_string(updated,"action_1_description"),"second") &&
       !strcmp(obs_data_get_string(updated,"action_2_description"),"first") &&
       obs_data_get_int(updated,"action_1_feature")==5 &&
       obs_data_get_int(updated,"action_2_feature")==1 &&
       obs_data_get_double(updated,"action_1_factor")==35.0 &&
       obs_data_get_double(updated,"action_2_factor")==18.0 &&
       !strcmp(obs_data_get_string(updated,"action_1_canvas"),"Custom") &&
       !obs_data_has_user_value(updated,"action_2_canvas");
   printf("REORDER_RESULT clicked=%d swapped=%d\n",(int)clicked,(int)swapped);
   obs_data_release(updated);
   if(props)obs_properties_destroy(props);
   obs_source_release(filter);obs_source_release(src);obs_shutdown();
   puts(swapped?"REORDER_PASS":"REORDER_FAIL");
   return swapped?0:13;
 }
 if(!strcmp(argv[3],"properties")){
   auto *props=obs_source_properties(filter);
   auto *group=obs_properties_get(props,"action_1_group");
   auto *action_group=group ? obs_property_group_content(group) : nullptr;
   const char *fields[]={
     "action_1_action","action_1_scene","action_1_sceneitem",
     "action_1_source","action_1_sceneitem_property","action_1_feature",
     "action_1_bounding_box","action_1_landmark_1","action_1_expression_1",
     "action_1_body_1","action_1_factor","action_1_threshold",
     "action_1_move_up","action_1_move_down","action_1_get_value"
   };
   bool ok=action_group!=nullptr;
   for(const char *field:fields){
     bool exists=action_group && obs_properties_get(action_group,field);
     printf("PROPERTIES_FIELD %s=%d\n",field,(int)exists);
     ok=ok && exists;
   }
   auto *selector=action_group ? obs_properties_get(action_group,"action_1_action"):nullptr;
   auto *show=obs_properties_get(props,"show");
   bool actions=selector && obs_property_list_item_count(selector)==5;
   printf("PROPERTIES_ACTIONS five=%d show=%d\n",(int)actions,(int)(show!=nullptr));
   ok=ok && actions && show;
   auto *ui_settings=obs_source_get_settings(filter);
   obs_data_set_int(ui_settings,"actions",2);
   auto *count_field=obs_properties_get(props,"actions");
   bool count_ok=count_field && obs_property_modified(count_field,ui_settings);
   auto *second=obs_properties_get(props,"action_2_group");
   bool second_visible=second && obs_property_visible(second);
   obs_data_set_int(ui_settings,"show",1);
   auto *show_field=obs_properties_get(props,"show");
   bool show_ok=show_field && obs_property_modified(show_field,ui_settings);
   bool second_hidden=second && !obs_property_visible(second);
   auto *up=action_group ? obs_properties_get(action_group,"action_1_move_up"):nullptr;
   auto *second_group=second ? obs_property_group_content(second):nullptr;
   auto *down=second_group ? obs_properties_get(second_group,"action_2_move_down"):nullptr;
   bool boundaries=up && down &&
       !obs_property_enabled(up) && !obs_property_enabled(down);
   obs_data_set_int(ui_settings,"actions",12);
   obs_data_set_int(ui_settings,"show",0);
   bool paginated=obs_property_modified(count_field,ui_settings) &&
       obs_data_get_int(ui_settings,"show")==1 &&
       second && !obs_property_visible(second);
   printf("PROPERTIES_PAGINATION boundaries=%d paginated=%d\n",
          (int)boundaries,(int)paginated);
   ok=ok && boundaries && paginated;
   obs_data_set_int(ui_settings,"actions",2);
   obs_property_modified(count_field,ui_settings);
   obs_data_set_int(ui_settings,"action_1_feature",5);
   auto *feature_field=action_group ? obs_properties_get(action_group,"action_1_feature"):nullptr;
   bool feature_ok=feature_field && obs_property_modified(feature_field,ui_settings);
   auto *body_field=action_group ? obs_properties_get(action_group,"action_1_body_1"):nullptr;
   bool body_visible=body_field && obs_property_visible(body_field);
   printf("PROPERTIES_CALLBACKS count=%d second_visible=%d show=%d second_hidden=%d feature=%d body_visible=%d\n",
          (int)count_ok,(int)second_visible,(int)show_ok,(int)second_hidden,
          (int)feature_ok,(int)body_visible);
   ok=ok && count_ok && second_visible && show_ok && second_hidden && feature_ok && body_visible;
   obs_source_update(filter,ui_settings);
   auto *value_field=action_group ? obs_properties_get(action_group,"action_1_get_value"):nullptr;
   bool value_clicked=value_field && obs_property_button_clicked(value_field,filter);
   bool unavailable=value_clicked &&
       !strcmp(obs_property_description(value_field),"No tracking value available");
   printf("PROPERTIES_NO_FAKE_BODY_VALUE clicked=%d unavailable=%d\n",
          (int)value_clicked,(int)unavailable);
   ok=ok && unavailable;
   obs_data_set_int(ui_settings,"action_1_feature",3);
   feature_ok=feature_field && obs_property_modified(feature_field,ui_settings);
   obs_data_set_int(ui_settings,"action_1_expression_property",1);
   auto *expr_prop=action_group ? obs_properties_get(action_group,"action_1_expression_property"):nullptr;
   bool expr_cb=expr_prop && obs_property_modified(expr_prop,ui_settings);
   auto *expr_2=action_group ? obs_properties_get(action_group,"action_1_expression_2"):nullptr;
   bool expr2_visible=expr_2 && obs_property_visible(expr_2);
   printf("PROPERTIES_EXPRESSION_VECTOR callback=%d second_visible=%d\n",
          (int)expr_cb,(int)expr2_visible);
   ok=ok && expr_cb && expr2_visible;
   obs_data_release(ui_settings);
   if(props) obs_properties_destroy(props);
   obs_source_release(filter);
   obs_source_release(src);
   obs_shutdown();
   puts(ok?"PROPERTIES_PASS":"PROPERTIES_FAIL");
   return ok?0:11;
 }
 if(!strcmp(argv[3],"body")){
   auto *props=obs_source_properties(filter);
   auto *group=obs_properties_get(props,"action_1_group");
   auto *sub=group ? obs_property_group_content(group):nullptr;
   auto *button=sub ? obs_properties_get(sub,"action_1_get_value"):nullptr;
   bool clicked=button && obs_property_button_clicked(button,filter);
   const char *label=button ? obs_property_description(button):"(missing)";
   printf("BODY_BEFORE_TRACKING clicked=%d value=%s\n",(int)clicked,label);
   bool unavailable=clicked && !strcmp(label,"No tracking value available");
   if(props)obs_properties_destroy(props);
   if(!unavailable)return 18;
 }
 puts("REPRO_ATTACH");
 obs_source_filter_add(src,filter);
 puts("REPRO_ATTACHED");
 if(argc>4 && !strcmp(argv[4],"render")){
   for(int i=0;i<24;i++){
    if(i==12 && !move_mode && face_pixels.size()==256*256*4 && active_fixture && active_fixture->texture){
      std::vector<uint8_t> blank(256*256*4,0);
      for(size_t px=3;px<blank.size();px+=4)blank[px]=255;
      obs_enter_graphics();
      gs_texture_set_image(active_fixture->texture,blank.data(),256*4,false);
      obs_leave_graphics();
      puts("REPRO_SWITCHED_TO_BLANK");
    }
    if(i==11 && pose_mode) {
      auto *props=obs_source_properties(filter);
      auto *group=obs_properties_get(props,"action_1_group");
      auto *sub=group ? obs_property_group_content(group):nullptr;
      auto *button=sub ? obs_properties_get(sub,"action_1_get_value"):nullptr;
      bool clicked=button && obs_property_button_clicked(button,filter);
      float value=button ? atof(obs_property_description(button)):0.0f;
      printf("POSE_QUATERNION_X clicked=%d value=%.3f\n",(int)clicked,value);
      bool valid=clicked && std::fabs(value)>=0.02f && std::fabs(value)<=1.0f;
      if(props)obs_properties_destroy(props);
      if(!valid)return 19;
      puts("POSE_PASS");
    }
    if(i==11 && move_mode) {
      struct vec2 pos{}; obs_sceneitem_get_pos(test_item,&pos);
      bool visible=obs_sceneitem_visible(test_item);
      printf("MOVE_ACTUAL position_x=%.3f position_y=%.3f visible=%d\n",pos.x,pos.y,(int)visible);
      if(expression_mode) {
        if(pos.x<=0.1 || pos.y<=0.1 || !visible) return 17;
        puts("EXPRESSION_VECTOR_PASS");
      } else if(attach_mode) {
        if(pos.x<60 || pos.x>210 || pos.y<40 || pos.y>220 || !visible) return 14;
        puts("ATTACH_PASS");
      } else if(pose_mode) {
   obs_data_set_int(settings,"actions",1);
   obs_data_set_int(settings,"action_1_feature",2);
   obs_data_set_int(settings,"action_1_pose",0); // quaternion X
 } else if(expression_mode) {
   obs_data_set_int(settings,"actions",1);
   obs_data_set_int(settings,"action_1_action",0);
   obs_data_set_string(settings,"action_1_scene","test scene");
   obs_data_set_string(settings,"action_1_sceneitem","overlay");
   obs_data_set_int(settings,"action_1_sceneitem_property",1); // vec2 position
   obs_data_set_int(settings,"action_1_feature",3); // expressions
   obs_data_set_int(settings,"action_1_expression_property",1); // vector
   obs_data_set_int(settings,"action_1_expression_1",5); // browOuterUpLeft
   obs_data_set_int(settings,"action_1_expression_2",14); // eyeLookInLeft
   obs_data_set_double(settings,"action_1_factor",10000);
   obs_data_set_double(settings,"action_1_factor2",10000);
 } else if(body_move_mode) {
        if(pos.x<10 || pos.x>500 || !visible) return 15;
        puts("BODY_MOVE_PASS");
      } else {
        if(pos.x < 60.0f || pos.x > 160.0f || visible) {
          puts("MOVE_FAIL");
          return 12;
        }
        auto *props=obs_source_properties(filter);
        auto *group=obs_properties_get(props,"action_1_group");
        auto *sub=group ? obs_property_group_content(group):nullptr;
        auto *button=sub ? obs_properties_get(sub,"action_1_get_value"):nullptr;
        bool clicked=button && obs_property_button_clicked(button,filter);
        float raw=button ? atof(obs_property_description(button)):-1;
        printf("MOVE_GET_VALUE clicked=%d value=%.3f\n",(int)clicked,raw);
        if(!clicked || raw<60 || raw>160) return 16;
        if(props)obs_properties_destroy(props);
        puts("MOVE_PASS");
      }
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
 if(test_scene) obs_scene_release(test_scene);
 puts("REPRO_SHUTDOWN");
 obs_shutdown();
 puts("REPRO_PASS");
 return 0;
}
