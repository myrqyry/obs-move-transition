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
static bool mp_move_log_error(struct mediapipe_move_info *filter, NvCV_Status nvErr, const char *function);
static void mp_move_feature_handle(struct mediapipe_move_info *filter, void *handle);
static void mp_move_landmarks(struct mediapipe_move_info *filter, void *handle, unsigned int size);
static bool mp_move_action_get_float(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, float *v);
static void mp_move_action_get_vec2(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, struct vec2 *value);
static void mp_move_update(void *data, obs_data_t *settings);
static void *mp_move_create(obs_data_t *settings, obs_source_t *context);
static void mp_move_destroy(void *data);
static void mp_move_actual_destroy(void *data);
static void mp_move_render(void *data);
static obs_properties_t *mp_move_properties(void *data);
static void mp_move_fill_body_list(obs_property_t *p);
static void mp_move_fill_landmark_list(obs_property_t *p);
static void mp_move_fill_expression_list(obs_property_t *p);
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
static void mp_move_defaults(obs_data_t *settings);

// MediaPipe Move Filter struct - mirrors nvidia_move_info but uses MediaPipe types
struct mediapipe_move_info {
	obs_source_t *source;

	DARRAY(struct nvidia_move_action) actions;

	// MediaPipe graph handles (one per feature type)
	void *face_graph;     // FaceLandmarker - covers bbox, landmarks, expressions, gaze
	void *pose_graph;     // PoseLandmarker - body keypoints

	char *last_error;

	bool got_new_frame;
	bool processing_stop;
	uint32_t width;
	uint32_t height;

	// Output arrays - parallel to nvidia_move_info
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

	gs_texrender_t *render;
	gs_texrender_t *render_unorm;
	enum gs_color_space space;

	gs_effect_t *effect;
	gs_eparam_t *image_param;
	gs_eparam_t *multiplier_param;
};

struct vec2 {
	float x;
	float y;
};

static const char *mp_move_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("MediaPipeMoveFilter");
}

static bool mp_move_log_error(struct mediapipe_move_info *filter, NvCV_Status nvErr, const char *function)
{
	if (nvErr == NVCV_SUCCESS)
		return false;
	const char *err_string = mediapipe_get_last_error();
	if (filter->last_error && strcmp(err_string, filter->last_error) == 0)
		return true;
	blog(LOG_ERROR, "[MediaPipe Move] Error in %s; error %i: %s", function, nvErr, err_string);
	bfree(filter->last_error);
	filter->last_error = bstrdup(err_string);
	return true;
}
