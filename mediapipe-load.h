// MediaPipe dynamic load shim - mirror pattern of nvar-load.h

#ifndef MEDIAPIPE_MOVE_FILTER_H
#define MEDIAPIPE_MOVE_FILTER_H

#include <stdbool.h>

// MediaPipe graph handles for different feature types
typedef struct mp_graph_t mp_graph_t;
typedef struct mp_face_landmarker_t mp_face_landmarker_t;
typedef struct mp_pose_landmarker_t mp_pose_landmarker_t;

// Point/rect types - mirrors NvAR_Point2f / NvAR_Point3f / NvAR_BBoxes
typedef struct {
	float x;
	float y;
} mp_point2f;

typedef struct {
	float x;
	float y;
	float z;
} mp_point3f;

typedef struct {
	float x;
	float y;
	float z;
	float w;
} mp_quaternion;

typedef struct {
	float x;
	float y;
	float width;
	float height;
	float confidence;
} mp_rect_t;

// Result containers
typedef struct {
	DARRAY(mp_point2f) landmarks;
	DARRAY(float) landmarks_visibility;
	DARRAY(float) blendshape_coefficients;
	mp_rect_t bounding_box;
} mp_face_result_t;

typedef struct {
	DARRAY(mp_point2f) keypoints;
	DARRAY(mp_point3f) keypoints_3d;
	DARRAY(float) visibility;
} mp_pose_result_t;

// Dynamic API table
typedef struct {
	bool loaded;
	void *handle;

	// MediaPipe Tasks API functions
	bool (*create_face_landmarker)(const char *model_path, mp_face_landmarker_t **landmarker);
	bool (*detect_face)(mp_face_landmarker_t *landmarker, const void *input, mp_face_result_t *result);
	void (*destroy_face_landmarker)(mp_face_landmarker_t *landmarker);
	bool (*create_pose_landmarker)(const char *model_path, mp_pose_landmarker_t **landmarker);
	bool (*detect_pose)(mp_pose_landmarker_t *landmarker, const void *input, mp_pose_result_t *result);
	void (*destroy_pose_landmarker)(mp_pose_landmarker_t *landmarker);

	// Utility functions
	const char *(*get_version)(void);
	const char *(*get_last_error)(void);
} mp_api_t;

// Global state
extern bool mediapipe_loaded;
extern mp_api_t mediapipe_api;

// Function prototypes
bool mediapipe_load_library(void);
void mediapipe_unload_library(void);
bool mediapipe_is_loaded(void);
mp_api_t *mediapipe_get_api(void);
const char *mediapipe_get_last_error(void);

#endif // MEDIAPIPE_MOVE_FILTER_H
