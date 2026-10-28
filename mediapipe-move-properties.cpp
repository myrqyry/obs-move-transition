// MediaPipe Move action editor. Settings keys deliberately match NVIDIA Move;
// tracking models/feature indices are adapted only where values actually exist.
#include "move-filter-actions.h"
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <obs-module.h>
#include <string>
#include <util/dstr.h>
#include <vector>

// Data accessors are implemented alongside the filter's private state.
extern obs_data_t *mp_move_ui_settings(void *data);
extern void mp_move_ui_apply_settings(void *data, obs_data_t *settings);
extern bool mp_move_ui_action_value(void *data, int action_number, char *out, size_t capacity);

namespace {
constexpr int MAX_ACTIONS = 40;
struct Choice {
  const char *label;
  int value;
};

static void key(char *out, size_t size, int i, const char *field) { snprintf(out, size, "action_%d_%s", i, field); }
static obs_property_t *field(obs_properties_t *props, int i, const char *suffix) {
  char buf[80];
  key(buf, sizeof buf, i, suffix);
  return obs_properties_get(props, buf);
}
static obs_property_t *choices(obs_properties_t *props, int i, const char *suffix, const char *label,
                               std::initializer_list<Choice> options) {
  char buf[80];
  key(buf, sizeof buf, i, suffix);
  obs_property_t *p = obs_properties_add_list(props, buf, label, OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
  for (const auto &o : options)
    obs_property_list_add_int(p, o.label, o.value);
  return p;
}
static obs_property_t *editable(obs_properties_t *props, int i, const char *suffix, const char *label) {
  char buf[80];
  key(buf, sizeof buf, i, suffix);
  return obs_properties_add_list(props, buf, label, OBS_COMBO_TYPE_EDITABLE, OBS_COMBO_FORMAT_STRING);
}
static void number(obs_properties_t *props, int i, const char *suffix, const char *label, double lo, double hi) {
  char buf[80];
  key(buf, sizeof buf, i, suffix);
  obs_properties_add_float(props, buf, label, lo, hi, 0.01);
}
static int selected(obs_data_t *settings, int i, const char *suffix) {
  char buf[80];
  key(buf, sizeof buf, i, suffix);
  return (int)obs_data_get_int(settings, buf);
}
static const char *selected_string(obs_data_t *settings, int i, const char *suffix) {
  char buf[80];
  key(buf, sizeof buf, i, suffix);
  return obs_data_get_string(settings, buf);
}
static void visible(obs_properties_t *props, int i, const char *suffix, bool value) {
  if (auto *p = field(props, i, suffix))
    obs_property_set_visible(p, value);
}
static bool source_name(void *arg, obs_source_t *source) {
  auto *p = static_cast<obs_property_t *>(arg);
  obs_property_list_add_string(p, obs_source_get_name(source), obs_source_get_name(source));
  return true;
}
static bool scene_item(obs_scene_t *, obs_sceneitem_t *item, void *arg) {
  auto *p = static_cast<obs_property_t *>(arg);
  obs_source_t *source = obs_sceneitem_get_source(item);
  if (source)
    obs_property_list_add_string(p, obs_source_get_name(source), obs_source_get_name(source));
  return true;
}
static void filter_name(obs_source_t *, obs_source_t *filter, void *arg) { source_name(arg, filter); }
static void populate_sceneitems(obs_properties_t *props, int i, obs_data_t *settings) {
  auto *p = field(props, i, "sceneitem");
  if (!p)
    return;
  obs_property_list_clear(p);
  const char *name = selected_string(settings, i, "scene");
  obs_source_t *src = obs_get_source_by_name(name);
  if (src) {
    obs_scene_t *scene = obs_scene_from_source(src);
    if (!scene)
      scene = obs_group_from_source(src);
    if (scene)
      obs_scene_enum_items(scene, scene_item, p);
    obs_source_release(src);
  }
}
static void populate_filters(obs_properties_t *props, int i, obs_data_t *settings) {
  auto *p = field(props, i, "filter");
  if (!p)
    return;
  obs_property_list_clear(p);
  obs_source_t *src = obs_get_source_by_name(selected_string(settings, i, "source"));
  if (src) {
    obs_source_enum_filters(src, filter_name, p);
    obs_source_release(src);
  }
}
static void populate_numeric_properties(obs_properties_t *props, int i, obs_data_t *settings) {
  auto *p = field(props, i, "property");
  if (!p)
    return;
  obs_property_list_clear(p);
  obs_source_t *src = obs_get_source_by_name(selected_string(settings, i, "source"));
  if (!src)
    return;
  const char *filter = selected_string(settings, i, "filter");
  obs_source_t *target = filter[0] ? obs_source_get_filter_by_name(src, filter) : nullptr;
  obs_properties_t *source_props = obs_source_properties(target ? target : src);
  if (source_props) {
    for (obs_property_t *it = obs_properties_first(source_props); it; obs_property_next(&it)) {
      enum obs_property_type t = obs_property_get_type(it);
      if (t == OBS_PROPERTY_INT || t == OBS_PROPERTY_FLOAT)
        obs_property_list_add_string(p, obs_property_description(it), obs_property_name(it));
    }
    obs_properties_destroy(source_props);
  }
  if (target)
    obs_source_release(target);
  obs_source_release(src);
}
static void refresh_details(obs_properties_t *props, int i, obs_data_t *settings) {
  const int action = selected(settings, i, "action");
  const int feature = selected(settings, i, "feature");
  const int lm = selected(settings, i, "landmark");
  const int expr = selected(settings, i, "expression_property");
  const int body = selected(settings, i, "body");
  const bool scene = action == ACTION_MOVE_SOURCE || action == ACTION_ATTACH_SOURCE || action == ACTION_SOURCE_VISIBILITY;
  const bool src = action == ACTION_MOVE_VALUE || action == ACTION_ENABLE_FILTER;
  const bool threshold = action == ACTION_ENABLE_FILTER || action == ACTION_SOURCE_VISIBILITY;
  const bool attach = action == ACTION_ATTACH_SOURCE;
  visible(props, i, "canvas", scene);
  visible(props, i, "scene", scene);
  visible(props, i, "sceneitem", scene);
  visible(props, i, "source", src);
  visible(props, i, "filter", src);
  visible(props, i, "property", action == ACTION_MOVE_VALUE);
  visible(props, i, "sceneitem_property", action == ACTION_MOVE_SOURCE);
  visible(props, i, "attach", attach);
  visible(props, i, "enable", threshold);
  visible(props, i, "threshold", threshold);
  visible(props, i, "feature", !attach);
  visible(props, i, "bounding_box", !attach && feature == FEATURE_BOUNDINGBOX);
  visible(props, i, "landmark", !attach && feature == FEATURE_LANDMARK);
  visible(props, i, "landmark_1", !attach && feature == FEATURE_LANDMARK);
  visible(props, i, "landmark_2",
          !attach && feature == FEATURE_LANDMARK &&
              (lm == FEATURE_LANDMARK_DISTANCE || lm == FEATURE_LANDMARK_DIFF || lm == FEATURE_LANDMARK_DIFF_X ||
               lm == FEATURE_LANDMARK_DIFF_Y || lm == FEATURE_LANDMARK_ROT));
  visible(props, i, "pose", !attach && feature == FEATURE_POSE);
  visible(props, i, "expression_property", !attach && feature == FEATURE_EXPRESSION);
  visible(props, i, "expression_1", !attach && feature == FEATURE_EXPRESSION);
  visible(props, i, "expression_2", !attach && feature == FEATURE_EXPRESSION && expr != FEATURE_EXPRESSION_SINGLE);
  visible(props, i, "body", !attach && feature == FEATURE_BODY);
  visible(props, i, "body_1", !attach && feature == FEATURE_BODY);
  visible(props, i, "body_2",
          !attach && feature == FEATURE_BODY &&
              (body == BODY_2D_DISTANCE || body == BODY_2D_DIFF || body == BODY_2D_DIFF_X || body == BODY_2D_DIFF_Y ||
               body == BODY_2D_ROT));
  visible(props, i, "required_confidence",
          !attach && (feature == FEATURE_BOUNDINGBOX || feature == FEATURE_LANDMARK || feature == FEATURE_BODY));
  visible(props, i, "factor", !threshold);
  visible(props, i, "factor2", !threshold);
  visible(props, i, "diff", !threshold);
  visible(props, i, "diff2", !threshold);
  visible(props, i, "easing", !threshold);
}
static int action_index(const char *name) {
  int i = 0;
  if (sscanf(name, "action_%d_", &i) != 1 || i < 1 || i > MAX_ACTIONS)
    return 0;
  return i;
}
static bool detail_changed(void *, obs_properties_t *props, obs_property_t *p, obs_data_t *settings) {
  int i = action_index(obs_property_name(p));
  if (!i)
    return false;
  refresh_details(props, i, settings);
  if (strstr(obs_property_name(p), "_scene") || strstr(obs_property_name(p), "_canvas"))
    populate_sceneitems(props, i, settings);
  if (strstr(obs_property_name(p), "_source")) {
    populate_filters(props, i, settings);
    populate_numeric_properties(props, i, settings);
  }
  if (strstr(obs_property_name(p), "_filter"))
    populate_numeric_properties(props, i, settings);
  return true;
}
static bool count_changed(void *, obs_properties_t *props, obs_property_t *, obs_data_t *settings) {
  int count = (int)obs_data_get_int(settings, "actions");
  if (count < 1)
    count = 1;
  if (count > MAX_ACTIONS)
    count = MAX_ACTIONS;
  int show = (int)obs_data_get_int(settings, "show");
  if (show > count || show < 0 || (show == 0 && count > 10)) {
    show = count > 10 ? 1 : 0;
    obs_data_set_int(settings, "show", show);
  }
  auto *list = obs_properties_get(props, "show");
  if (list) {
    obs_property_list_clear(list);
    obs_property_list_add_int(list, "All actions", 0);
    for (int i = 1; i <= count; ++i) {
      char n[80], label[128];
      key(n, sizeof n, i, "description");
      const char *description = obs_data_get_string(settings, n);
      if (description[0])
        snprintf(label, sizeof label, "%d: %.110s", i, description);
      else
        snprintf(label, sizeof label, "Action %d", i);
      obs_property_list_add_int(list, label, i);
    }
  }
  for (int i = 1; i <= MAX_ACTIONS; ++i) {
    char n[80];
    key(n, sizeof n, i, "group");
    if (auto *group = obs_properties_get(props, n)) {
      obs_property_set_visible(group, i <= count && (show == 0 || show == i));
      auto *sub = obs_property_group_content(group);
      if (sub) {
        if (auto *up = field(sub, i, "move_up"))
          obs_property_set_enabled(up, i > 1 && i <= count);
        if (auto *down = field(sub, i, "move_down"))
          obs_property_set_enabled(down, i < count);
      }
    }
  }
  return true;
}
struct SavedSetting {
  std::string suffix;
  enum obs_data_type type = OBS_DATA_NULL;
  enum obs_data_number_type number_type = OBS_DATA_NUM_INVALID;
  std::string text;
  long long integer = 0;
  double real = 0;
  bool boolean = false;
  obs_data_t *object = nullptr;
  obs_data_array_t *array = nullptr;
};
static void put_saved(obs_data_t *settings, const std::string &key, const SavedSetting &v) {
  switch (v.type) {
  case OBS_DATA_STRING:
    obs_data_set_string(settings, key.c_str(), v.text.c_str());
    break;
  case OBS_DATA_NUMBER:
    if (v.number_type == OBS_DATA_NUM_INT)
      obs_data_set_int(settings, key.c_str(), v.integer);
    else
      obs_data_set_double(settings, key.c_str(), v.real);
    break;
  case OBS_DATA_BOOLEAN:
    obs_data_set_bool(settings, key.c_str(), v.boolean);
    break;
  case OBS_DATA_OBJECT:
    obs_data_set_obj(settings, key.c_str(), v.object);
    break;
  case OBS_DATA_ARRAY:
    obs_data_set_array(settings, key.c_str(), v.array);
    break;
  default:
    break;
  }
}
static void release_saved(std::vector<SavedSetting> &entries) {
  for (auto &v : entries) {
    if (v.object)
      obs_data_release(v.object);
    if (v.array)
      obs_data_array_release(v.array);
  }
}
static bool reorder_action(void *data, int current, int direction) {
  if (!data)
    return false;
  obs_data_t *settings = mp_move_ui_settings(data);
  if (!settings)
    return false;
  int count = (int)obs_data_get_int(settings, "actions");
  int target = current + direction;
  if (current < 1 || target < 1 || target > count) {
    obs_data_release(settings);
    return false;
  }
  char first[32], second[32];
  snprintf(first, sizeof first, "action_%d_", current);
  snprintf(second, sizeof second, "action_%d_", target);
  const std::string a = first, b = second;
  std::vector<SavedSetting> left, right;
  for (obs_data_item_t *item = obs_data_first(settings); item; obs_data_item_next(&item)) {
    const std::string name = obs_data_item_get_name(item);
    bool is_left = name.compare(0, a.size(), a) == 0;
    bool is_right = name.compare(0, b.size(), b) == 0;
    if ((!is_left && !is_right) || !obs_data_item_has_user_value(item))
      continue;
    SavedSetting value;
    value.suffix = name.substr(is_left ? a.size() : b.size());
    value.type = obs_data_item_gettype(item);
    value.number_type = obs_data_item_numtype(item);
    switch (value.type) {
    case OBS_DATA_STRING:
      value.text = obs_data_item_get_string(item);
      break;
    case OBS_DATA_NUMBER:
      value.integer = obs_data_item_get_int(item);
      value.real = obs_data_item_get_double(item);
      break;
    case OBS_DATA_BOOLEAN:
      value.boolean = obs_data_item_get_bool(item);
      break;
    case OBS_DATA_OBJECT:
      value.object = obs_data_item_get_obj(item);
      break;
    case OBS_DATA_ARRAY:
      value.array = obs_data_item_get_array(item);
      break;
    default:
      break;
    }
    (is_left ? left : right).push_back(std::move(value));
  }
  for (const auto &v : left)
    obs_data_unset_user_value(settings, (a + v.suffix).c_str());
  for (const auto &v : right)
    obs_data_unset_user_value(settings, (b + v.suffix).c_str());
  for (const auto &v : left)
    put_saved(settings, b + v.suffix, v);
  for (const auto &v : right)
    put_saved(settings, a + v.suffix, v);
  release_saved(left);
  release_saved(right);
  mp_move_ui_apply_settings(data, settings);
  obs_data_release(settings);
  return true;
}
static bool move_up(obs_properties_t *, obs_property_t *p, void *data) {
  return reorder_action(data, action_index(obs_property_name(p)), -1);
}
static bool move_down(obs_properties_t *, obs_property_t *p, void *data) {
  return reorder_action(data, action_index(obs_property_name(p)), 1);
}
static bool get_value(obs_properties_t *, obs_property_t *p, void *data) {
  char buf[100];
  if (mp_move_ui_action_value(data, action_index(obs_property_name(p)), buf, sizeof buf))
    obs_property_set_description(p, buf);
  else
    obs_property_set_description(p, "No tracking value available");
  return true;
}

static void add_expressions(obs_property_t *p) {
  // MediaPipe's canonical blendshape order; indices in settings remain 1-based.
  static const char *names[] = {
      "_neutral",         "browDownLeft",      "browDownRight",    "browInnerUp",     "browOuterUpLeft",    "browOuterUpRight",
      "cheekPuff",        "cheekSquintLeft",   "cheekSquintRight", "eyeBlinkLeft",    "eyeBlinkRight",      "eyeLookDownLeft",
      "eyeLookDownRight", "eyeLookInLeft",     "eyeLookInRight",   "eyeLookOutLeft",  "eyeLookOutRight",    "eyeLookUpLeft",
      "eyeLookUpRight",   "eyeSquintLeft",     "eyeSquintRight",   "eyeWideLeft",     "eyeWideRight",       "jawForward",
      "jawLeft",          "jawOpen",           "jawRight",         "mouthClose",      "mouthDimpleLeft",    "mouthDimpleRight",
      "mouthFrownLeft",   "mouthFrownRight",   "mouthFunnel",      "mouthLeft",       "mouthLowerDownLeft", "mouthLowerDownRight",
      "mouthPressLeft",   "mouthPressRight",   "mouthPucker",      "mouthRight",      "mouthRollLower",     "mouthRollUpper",
      "mouthShrugLower",  "mouthShrugUpper",   "mouthSmileLeft",   "mouthSmileRight", "mouthStretchLeft",   "mouthStretchRight",
      "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",    "noseSneerRight"};
  for (size_t j = 0; j < sizeof(names) / sizeof(names[0]); ++j)
    obs_property_list_add_int(p, names[j], (long long)j + 1);
}
static void add_body(obs_property_t *p) {
  static const char *names[] = {"nose",          "leftEyeInner",  "leftEye",       "leftEyeOuter", "rightEyeInner", "rightEye",
                                "rightEyeOuter", "leftEar",       "rightEar",      "mouthLeft",    "mouthRight",    "leftShoulder",
                                "rightShoulder", "leftElbow",     "rightElbow",    "leftWrist",    "rightWrist",    "leftPinky",
                                "rightPinky",    "leftIndex",     "rightIndex",    "leftThumb",    "rightThumb",    "leftHip",
                                "rightHip",      "leftKnee",      "rightKnee",     "leftAnkle",    "rightAnkle",    "leftHeel",
                                "rightHeel",     "leftFootIndex", "rightFootIndex"};
  for (size_t j = 0; j < sizeof(names) / sizeof(names[0]); ++j)
    obs_property_list_add_int(p, names[j], (long long)j);
}
static void on_change(obs_property_t *p) { obs_property_set_modified_callback2(p, detail_changed, nullptr); }
static void add_action(obs_properties_t *props, int i, void *data) {
  obs_properties_t *group = obs_properties_create();
  char buf[100];
  key(buf, sizeof buf, i, "disabled");
  obs_properties_add_bool(group, buf, "Disabled");
  key(buf, sizeof buf, i, "description");
  obs_properties_add_text(group, buf, "Action description", OBS_TEXT_DEFAULT);
  obs_property_t *p = choices(group, i, "action", "Action",
                              {{"Move source", ACTION_MOVE_SOURCE},
                               {"Move value", ACTION_MOVE_VALUE},
                               {"Enable filter", ACTION_ENABLE_FILTER},
                               {"Source visibility", ACTION_SOURCE_VISIBILITY},
                               {"Attach source", ACTION_ATTACH_SOURCE}});
  on_change(p);
  p = choices(group, i, "attach", "Attach",
              {{"Eyes", ATTACH_EYES},
               {"Left eye", ATTACH_LEFT_EYE},
               {"Right eye", ATTACH_RIGHT_EYE},
               {"Eyebrows", ATTACH_EYEBROWS},
               {"Left eyebrow", ATTACH_LEFT_EYEBROW},
               {"Right eyebrow", ATTACH_RIGHT_EYEBROW},
               {"Ears", ATTACH_EARS},
               {"Left ear", ATTACH_LEFT_EAR},
               {"Right ear", ATTACH_RIGHT_EAR},
               {"Nose", ATTACH_NOSE},
               {"Mouth", ATTACH_MOUTH},
               {"Upper lip", ATTACH_UPPER_LIP},
               {"Lower lip", ATTACH_LOWER_LIP},
               {"Chin", ATTACH_CHIN},
               {"Jaw", ATTACH_JAW},
               {"Forehead", ATTACH_FOREHEAD}});
  p = editable(group, i, "canvas", "Canvas");
  p = editable(group, i, "scene", "Scene");
  obs_enum_scenes(source_name, p);
  on_change(p);
  p = editable(group, i, "sceneitem", "Source in scene");
  p = editable(group, i, "source", "Source");
  obs_enum_sources(source_name, p);
  on_change(p);
  p = editable(group, i, "filter", "Filter (optional)");
  on_change(p);
  p = editable(group, i, "property", "Numeric property");
  p = choices(group, i, "sceneitem_property", "Scene item property",
              {{"Position", SCENEITEM_PROPERTY_POS},
               {"Position X", SCENEITEM_PROPERTY_POSX},
               {"Position Y", SCENEITEM_PROPERTY_POSY},
               {"Scale", SCENEITEM_PROPERTY_SCALE},
               {"Scale X", SCENEITEM_PROPERTY_SCALEX},
               {"Scale Y", SCENEITEM_PROPERTY_SCALEY},
               {"Rotation", SCENEITEM_PROPERTY_ROT},
               {"Crop left", SCENEITEM_PROPERTY_CROP_LEFT},
               {"Crop right", SCENEITEM_PROPERTY_CROP_RIGHT},
               {"Crop top", SCENEITEM_PROPERTY_CROP_TOP},
               {"Crop bottom", SCENEITEM_PROPERTY_CROP_BOTTOM}});
  p = choices(group, i, "enable", "Threshold action",
              {{"Enable over", FEATURE_THRESHOLD_ENABLE_OVER},
               {"Enable under", FEATURE_THRESHOLD_ENABLE_UNDER},
               {"Disable over", FEATURE_THRESHOLD_DISABLE_OVER},
               {"Disable under", FEATURE_THRESHOLD_DISABLE_UNDER},
               {"Enable over / disable under", FEATURE_THRESHOLD_ENABLE_OVER_DISABLE_UNDER},
               {"Enable under / disable over", FEATURE_THRESHOLD_ENABLE_UNDER_DISABLE_OVER}});
  p = choices(group, i, "feature", "Tracking feature",
              {{"Face bounding box", FEATURE_BOUNDINGBOX},
               {"Face landmark", FEATURE_LANDMARK},
               {"Head transformation", FEATURE_POSE},
               {"Face expression", FEATURE_EXPRESSION},
               {"Body landmark", FEATURE_BODY}});
  on_change(p);
  p = choices(group, i, "bounding_box", "Bounding box coordinate",
              {{"Left", FEATURE_BOUNDINGBOX_LEFT},
               {"Center X", FEATURE_BOUNDINGBOX_HORIZONTAL_CENTER},
               {"Right", FEATURE_BOUNDINGBOX_RIGHT},
               {"Top", FEATURE_BOUNDINGBOX_TOP},
               {"Center Y", FEATURE_BOUNDINGBOX_VERTICAL_CENTER},
               {"Bottom", FEATURE_BOUNDINGBOX_BOTOM},
               {"Width", FEATURE_BOUNDINGBOX_WIDTH},
               {"Height", FEATURE_BOUNDINGBOX_HEIGHT},
               {"Top left", FEATURE_BOUNDINGBOX_TOP_LEFT},
               {"Top center", FEATURE_BOUNDINGBOX_TOP_CENTER},
               {"Top right", FEATURE_BOUNDINGBOX_TOP_RIGHT},
               {"Center right", FEATURE_BOUNDINGBOX_CENTER_RIGHT},
               {"Bottom right", FEATURE_BOUNDINGBOX_BOTTOM_RIGHT},
               {"Bottom center", FEATURE_BOUNDINGBOX_BOTTOM_CENTER},
               {"Bottom left", FEATURE_BOUNDINGBOX_BOTTOM_LEFT},
               {"Center left", FEATURE_BOUNDINGBOX_CENTER_LEFT},
               {"Center", FEATURE_BOUNDINGBOX_CENTER},
               {"Size", FEATURE_BOUNDINGBOX_SIZE}});
  p = choices(group, i, "landmark", "Face landmark property",
              {{"X", FEATURE_LANDMARK_X},
               {"Y", FEATURE_LANDMARK_Y},
               {"Confidence", FEATURE_LANDMARK_CONFIDENCE},
               {"Distance", FEATURE_LANDMARK_DISTANCE},
               {"Difference X", FEATURE_LANDMARK_DIFF_X},
               {"Difference Y", FEATURE_LANDMARK_DIFF_Y},
               {"Rotation", FEATURE_LANDMARK_ROT},
               {"Difference", FEATURE_LANDMARK_DIFF},
               {"Position", FEATURE_LANDMARK_POS}});
  on_change(p);
  key(buf, sizeof buf, i, "landmark_1");
  obs_properties_add_int_slider(group, buf, "Face landmark 1 (1-478)", 1, 478, 1);
  key(buf, sizeof buf, i, "landmark_2");
  obs_properties_add_int_slider(group, buf, "Face landmark 2 (1-478)", 1, 478, 1);
  p = choices(group, i, "pose", "Head transformation component",
              {{"X", FEATURE_POSE_X}, {"Y", FEATURE_POSE_Y}, {"Z", FEATURE_POSE_Z}, {"W", FEATURE_POSE_W}});
  p = choices(group, i, "expression_property", "Expression property",
              {{"Single", FEATURE_EXPRESSION_SINGLE},
               {"Vector", FEATURE_EXPRESSION_VECTOR},
               {"Add", FEATURE_EXPRESSION_ADD},
               {"Subtract", FEATURE_EXPRESSION_SUBSTRACT},
               {"Distance", FEATURE_EXPRESSION_DISTANCE},
               {"Average", FEATURE_EXPRESSION_AVG}});
  on_change(p);
  key(buf, sizeof buf, i, "expression_1");
  p = obs_properties_add_list(group, buf, "Expression 1", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
  add_expressions(p);
  key(buf, sizeof buf, i, "expression_2");
  p = obs_properties_add_list(group, buf, "Expression 2", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
  add_expressions(p);
  p = choices(group, i, "body", "Body property",
              {{"Confidence", BODY_CONFIDENCE},
               {"2D X", BODY_2D_POSX},
               {"2D Y", BODY_2D_POSY},
               {"2D distance", BODY_2D_DISTANCE},
               {"2D rotation", BODY_2D_ROT},
               {"2D diff X", BODY_2D_DIFF_X},
               {"2D diff Y", BODY_2D_DIFF_Y},
               {"2D difference", BODY_2D_DIFF},
               {"2D position", BODY_2D_POS},
               {"3D X", BODY_3D_POSX},
               {"3D Y", BODY_3D_POSY},
               {"3D Z", BODY_3D_POSZ}});
  on_change(p);
  key(buf, sizeof buf, i, "body_1");
  p = obs_properties_add_list(group, buf, "Body landmark 1", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
  add_body(p);
  key(buf, sizeof buf, i, "body_2");
  p = obs_properties_add_list(group, buf, "Body landmark 2", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
  add_body(p);
  number(group, i, "required_confidence", "Required confidence", 0, 1);
  number(group, i, "factor", "Factor X (%)", -1000000, 1000000);
  number(group, i, "factor2", "Factor Y (%)", -1000000, 1000000);
  number(group, i, "diff", "Offset X", -1000000, 1000000);
  number(group, i, "diff2", "Offset Y", -1000000, 1000000);
  number(group, i, "threshold", "Threshold", -1000000, 1000000);
  key(buf, sizeof buf, i, "easing");
  obs_properties_add_float_slider(group, buf, "Easing (%)", 0, 99, 0.01);
  key(buf, sizeof buf, i, "get_value");
  obs_properties_add_button2(group, buf, "Get current value", get_value, data);
  key(buf, sizeof buf, i, "move_up");
  obs_properties_add_button2(group, buf, "Move up", move_up, data);
  key(buf, sizeof buf, i, "move_down");
  obs_properties_add_button2(group, buf, "Move down", move_down, data);
  key(buf, sizeof buf, i, "group");
  char label[70];
  snprintf(label, sizeof label, "Action %d", i);
  obs_properties_add_group(props, buf, label, OBS_GROUP_NORMAL, group);
}
} // namespace

obs_properties_t *mp_move_properties(void *data) {
  obs_properties_t *props = obs_properties_create();
  auto *p = obs_properties_add_int(props, "actions", "Actions", 1, MAX_ACTIONS, 1);
  obs_property_set_modified_callback2(p, count_changed, nullptr);
  p = obs_properties_add_list(props, "show", "Show", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
  obs_property_set_modified_callback2(p, count_changed, nullptr);
  for (int i = 1; i <= MAX_ACTIONS; ++i)
    add_action(props, i, data);
  obs_properties_add_float_slider(props, "face_detection_confidence", "Face detection confidence", 0, 1, 0.01);
  obs_properties_add_float_slider(props, "face_presence_confidence", "Face presence confidence", 0, 1, 0.01);
  obs_properties_add_float_slider(props, "face_tracking_confidence", "Face tracking confidence", 0, 1, 0.01);
  // Use current settings for initial group visibility and populated selection lists.
  // Only the UI observes this data; actual action parsing remains in the filter.
  extern obs_data_t *mp_move_ui_settings(void *);
  obs_data_t *settings = mp_move_ui_settings(data);
  if (!settings) {
    settings = obs_data_create();
    obs_data_set_default_int(settings, "actions", 1);
  }
  if (settings) {
    count_changed(nullptr, props, nullptr, settings);
    int count = (int)obs_data_get_int(settings, "actions");
    for (int i = 1; i <= count && i <= MAX_ACTIONS; ++i) {
      char group_name[80];
      key(group_name, sizeof group_name, i, "group");
      auto *group = obs_properties_get(props, group_name);
      auto *sub = group ? obs_property_group_content(group) : nullptr;
      if (!sub)
        continue;
      refresh_details(sub, i, settings);
      populate_sceneitems(sub, i, settings);
      populate_filters(sub, i, settings);
      populate_numeric_properties(sub, i, settings);
    }
    obs_data_release(settings);
  }
  return props;
}
