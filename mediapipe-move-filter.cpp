// MediaPipe Move Filter - mirror of nvidia-move-filter.c with MediaPipe C++ backend
// Uses MediaPipe Tasks C++ API (FaceLandmarker, PoseLandmarker) via runtime loading

#include "mediapipe-load.h"
#include "move-filter-actions.h"
#include <obs-module.h>
#include <util/threading.h>
#include <util/dstr.h>
#include "easing.h"
#include "move-transition.h"

#define BBOXES_COUNT 25
#define MAX_ACTIONS 40

// Forward declarations matching nvidia-move-filter.c pattern
static const char *mp_move_name(void *unused);
static bool mp_move_log_error(struct mediapipe_move_info *filter, bool success, const char *function);
static bool mp_move_action_get_float(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, float *v);
static void mp_move_action_get_vec2(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, struct vec2 *value);
static void mp_move_update(void *data, obs_data_t *settings);
static void *mp_move_create(obs_data_t *settings, obs_source_t *context);
static void mp_move_destroy(void *data);
static void mp_move_actual_destroy(void *data);
static void mp_move_render(void *data, gs_effect_t *effect);
static obs_properties_t *mp_move_properties(void *data);
static void mp_move_fill_body_list(obs_property_t *p);
static void mp_move_fill_landmark_list(obs_property_t *p);
static void mp_move_fill_expression_list(obs_property_t *p);
static void mp_move_tick(void *data, float t);
static enum gs_color_space mp_move_get_color_space(void *data, size_t count, const enum gs_color_space *preferred_spaces);
static void mp_move_defaults(obs_data_t *settings);
static void swap_setting(obs_data_t *settings, char *setting1, char *setting2);
static void swap_action(obs_data_t *settings, long long a, long long b);
static bool mp_move_move_up_clicked(obs_properties_t *props, obs_property_t *property, void *data);
static bool mp_move_move_down_clicked(obs_properties_t *props, obs_property_t *property, void *data);
static bool mp_move_get_value_clicked(obs_properties_t *props, obs_property_t *property, void *data);
static bool mp_move_feature_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool mp_move_landmark_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool mp_move_body_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool mp_move_expression_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool mp_move_actions_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
void mp_move_register(void);

// MediaPipe Move Filter struct
struct mediapipe_move_info {
	obs_source_t *source;
	DARRAY(struct nvidia_move_action) actions;
	void *face_graph;
	void *pose_graph;
	char *last_error;
	bool got_new_frame;
	bool processing_stop;
	bool target_valid;
	bool initial_render;
	bool processed_frame;
	uint32_t width;
	uint32_t height;

	DARRAY(float) landmarks_confidence;
	DARRAY(mp_point2f) landmarks;
	DARRAY(mp_point2f) gaze_output_landmarks;
	DARRAY(float) bboxes_confidence;
	mp_rect_t bbox;
	DARRAY(float) expressions;
	float gaze_angles[2];
	DARRAY(float) keypoints_confidence;
	DARRAY(mp_point2f) keypoints;
	DARRAY(mp_point3f) keypoints3D;
	DARRAY(mp_quaternion) joint_angles;

	gs_texrender_t *render;
	gs_texrender_t *render_unorm;
	enum gs_color_space space;

	gs_effect_t *effect;
	gs_eparam_t *image_param;
	gs_eparam_t *multiplier_param;
};

static const char *mp_move_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("MediaPipeMoveFilter");
}

static bool mp_move_log_error(struct mediapipe_move_info *filter, bool success, const char *function)
{
	if (success)
		return false;
	const char *err_string = mediapipe_api.get_last_error ? mediapipe_api.get_last_error() : "unknown error";
	if (filter->last_error && strcmp(err_string, filter->last_error) == 0)
		return true;
	blog(LOG_ERROR, "[MediaPipe Move] Error in %s: %s", function, err_string);
	bfree(filter->last_error);
	filter->last_error = bstrdup(err_string);
	return true;
}

static bool mp_check_model(const char *name)
{
	char *path = obs_module_file(name);
	bool exists = os_file_exists(path);
	if (!exists) {
		blog(LOG_ERROR, "[MediaPipe] Model file missing: %s. Please download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", name);
	}
	bfree(path);
	return exists;
}

static void *mp_move_create(obs_data_t *settings, obs_source_t *context)
{
	if (!mediapipe_is_loaded()) return NULL;
	if (!mp_check_model("face_landmarker.task")) return NULL;
	if (!mp_check_model("pose_landmarker.task")) return NULL;

	struct mediapipe_move_info *filter = (struct mediapipe_move_info *)bzalloc(sizeof(*filter));
	filter->source = context;

	mp_api_t *api = mediapipe_get_api();
	char *face_model_path = obs_module_file("face_landmarker.task");
	char *pose_model_path = obs_module_file("pose_landmarker.task");
	if (!api->create_face_landmarker(face_model_path, (mp_face_landmarker_t **)&filter->face_graph)) {
		blog(LOG_ERROR, "[MediaPipe] Failed to create face landmarker");
	}
	if (!api->create_pose_landmarker(pose_model_path, (mp_pose_landmarker_t **)&filter->pose_graph)) {
		blog(LOG_ERROR, "[MediaPipe] Failed to create pose landmarker");
	}
	bfree(face_model_path);
	bfree(pose_model_path);

	char *effect_path = obs_module_file("effects/unorm.effect");
	obs_enter_graphics();
	filter->effect = gs_effect_create_from_file(effect_path, NULL);
	bfree(effect_path);
	if (filter->effect) {
		filter->image_param = gs_effect_get_param_by_name(filter->effect, "image");
		filter->multiplier_param = gs_effect_get_param_by_name(filter->effect, "multiplier");
	}
	obs_leave_graphics();

	filter->space = GS_CS_SRGB;
	obs_source_update(context, settings);
	return filter;
}

static void mp_move_destroy(void *data)
{
	struct mediapipe_move_info *filter = (struct mediapipe_move_info *)data;
	da_free(filter->actions);
	obs_queue_task(OBS_TASK_GRAPHICS, mp_move_actual_destroy, data, false);
}

static void mp_move_actual_destroy(void *data)
{
	struct mediapipe_move_info *filter = (struct mediapipe_move_info *)data;
	bfree(filter->last_error);
	bfree(filter);
}

static void mp_move_render(void *data, gs_effect_t *effect) {}
static void mp_move_tick(void *data, float t) {}
static enum gs_color_space mp_move_get_color_space(void *data, size_t count, const enum gs_color_space *preferred_spaces) { return GS_CS_SRGB; }
static void mp_move_defaults(obs_data_t *settings) { obs_data_set_default_int(settings, "actions", 1); }
static obs_properties_t *mp_move_properties(void *data) { return obs_properties_create(); }
static void mp_move_fill_body_list(obs_property_t *p) {}
static void mp_move_fill_landmark_list(obs_property_t *p) {}
static void mp_move_fill_expression_list(obs_property_t *p) {}
static void swap_setting(obs_data_t *settings, char *setting1, char *setting2) {}
static void swap_action(obs_data_t *settings, long long a, long long b) {}
static bool mp_move_move_up_clicked(obs_properties_t *props, obs_property_t *property, void *data) { return true; }
static bool mp_move_move_down_clicked(obs_properties_t *props, obs_property_t *property, void *data) { return true; }
static bool mp_move_get_value_clicked(obs_properties_t *props, obs_property_t *property, void *data) { return true; }
static bool mp_move_feature_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { return true; }
static bool mp_move_landmark_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { return true; }
static bool mp_move_body_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { return true; }
static bool mp_move_expression_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { return true; }
static bool mp_move_actions_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { return true; }
static void mp_move_update(void *data, obs_data_t *settings) {}
static bool mp_move_action_get_float(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, float *v) { *v = 0.0f; return false; }
static void mp_move_action_get_vec2(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, struct vec2 *value) { value->x = 0.0f; value->y = 0.0f; }

struct obs_source_info mediapipe_move_filter = {
	.id = "mp_move_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB,
	.get_name = mp_move_name,
	.create = mp_move_create,
	.destroy = mp_move_destroy,
	.get_defaults = mp_move_defaults,
	.get_properties = mp_move_properties,
	.update = mp_move_update,
	.load = mp_move_update,
	.video_render = mp_move_render,
	.video_tick = mp_move_tick,
	.video_get_color_space = mp_move_get_color_space,
};

void mp_move_register(void)
{
	obs_register_source(&mediapipe_move_filter);
}
