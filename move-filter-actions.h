// Move filter actions - shared constants and structs
// This file contains all ACTION_*, FEATURE_*, ATTACH_*, and SCENEITEM_PROPERTY_*
// defines used by both nvidia-move-filter.c and mediapipe-move-filter.cpp.
// Guards against duplicate inclusion to allow both nvidia and mediapipe filters
// to share the same settings and action definitions while keeping separate dynamics.

#ifndef MOVE_FILTER_ACTIONS_H
#define MOVE_FILTER_ACTIONS_H

// Action types for filter operations - mirror NVIDIA pattern exactly
#define ACTION_MOVE_SOURCE          0
#define ACTION_MOVE_VALUE           1
#define ACTION_ENABLE_FILTER        2
#define ACTION_SOURCE_VISIBILITY    3
#define ACTION_ATTACH_SOURCE        4

// Attachment locations for sources - same indices across both implementations
#define ATTACH_EYES                 0
#define ATTACH_LEFT_EYE             1
#define ATTACH_RIGHT_EYE            2
#define ATTACH_EYEBROWS             3
#define ATTACH_LEFT_EYEBROW         4
#define ATTACH_RIGHT_EYEBROW        5
#define ATTACH_EARS                 6
#define ATTACH_LEFT_EAR             7
#define ATTACH_RIGHT_EAR            8
#define ATTACH_NOSE                 9
#define ATTACH_MOUTH                10
#define ATTACH_UPPER_LIP            11
#define ATTACH_LOWER_LIP            12
#define ATTACH_CHIN                 13
#define ATTACH_JAW                  14
#define ATTACH_FOREHEAD            15

// Scene item property identifiers - sequential indices for serialization
// Must match exactly between nvidia-move-filter.c and mediapipe-move-filter.cpp
// to allow settings interchange when the user switches backends.
#define SCENEITEM_PROPERTY_ALL       0
#define SCENEITEM_PROPERTY_POS       1
#define SCENEITEM_PROPERTY_POSX      2
#define SCENEITEM_PROPERTY_POSY      3
#define SCENEITEM_PROPERTY_SCALE     4
#define SCENEITEM_PROPERTY_SCALEX    5
#define SCENEITEM_PROPERTY_SCALEY    6
#define SCENEITEM_PROPERTY_ROT       7
#define SCENEITEM_PROPERTY_CROP_LEFT  8
#define SCENEITEM_PROPERTY_CROP_RIGHT 9
#define SCENEITEM_PROPERTY_CROP_BOTTOM 10
#define SCENEITEM_PROPERTY_CROP_TOP   11

// Feature identifiers - trackable elements in both MediaPipe and NVIDIA AR SDKs
// Feature indices are serialized and controlled by user settings. These are shared
// across both nvidia-move-filter.c and mediapipe-move-filter.cpp to ensure identical
// UI content and algorithm behavior across the two backends.
#define FEATURE_BOUNDINGBOX         0
#define FEATURE_LANDMARK             1
#define FEATURE_POSE                2
#define FEATURE_EXPRESSION          3
#define FEATURE_GAZE                4
#define FEATURE_BODY                5

// Bounding box sub-features - normalize property naming across backends
// Mirror NVIDIA's naming convention to ensure cross-backend consistency.
// BOTOM field renamed to BOW for better consistency and clarity.
#define FEATURE_BOUNDINGBOX_LEFT             0
#define FEATURE_BOUNDINGBOX_HORIZONTAL_CENTER 1
#define FEATURE_BOUNDINGBOX_RIGHT             2
#define FEATURE_BOUNDINGBOX_WIDTH             3
#define FEATURE_BOUNDINGBOX_TOP              4
#define FEATURE_BOUNDINGBOX_VERTICAL_CENTER  5
#define FEATURE_BOUNDINGBOX_BOW               6
#define FEATURE_BOUNDINGBOX_HEIGHT           7
#define FEATURE_BOUNDINGBOX_TOP_LEFT           8
#define FEATURE_BOUNDINGBOX_TOP_CENTER        9
#define FEATURE_BOUNDINGBOX_TOP_RIGHT        10
#define FEATURE_BOUNDINGBOX_CENTER_RIGHT      11
#define FEATURE_BOUNDINGBOX_BOTTOM_RIGHT     12
#define FEATURE_BOUNDINGBOX_BOTTOM_CENTER    13
#define FEATURE_BOUNDINGBOX_BOTTOM_LEFT      14
#define FEATURE_BOUNDINGBOX_CENTER_LEFT      15
#define FEATURE_BOUNDINGBOX_CENTER           16
#define FEATURE_BOUNDINGBOX_SIZE             17

// Landmark sub-features - streamlined property identifiers for keypoint data
// Includes confidence score and keypoint positioning information for comprehensive
// landmark analysis and tracking capabilities.
#define FEATURE_LANDMARK_X                 0
#define FEATURE_LANDMARK_Y                 1
#define FEATURE_LANDMARK_CONFIDENCE         2
#define FEATURE_LANDMARK_DISTANCE          3
#define FEATURE_LANDMARK_DIFF_X            4
#define FEATURE_LANDMARK_DIFF_Y            5
#define FEATURE_LANDMARK_ROT               6
#define FEATURE_LANDMARK_DIFF              7
#define FEATURE_LANDMARK_POS               8

// Threshold for landmark visibility - hide points below specified reliability
#define FEATURE_THRESHOLD_NONE                      0
#define FEATURE_THRESHOLD_ENABLE_OVER                1
#define FEATURE_THRESHOLD_ENABLE_UNDER               2
#define FEATURE_THRESHOLD_DISABLE_OVER               3
#define FEATURE_THRESHOLD_DISABLE_UNDER              4
#define FEATURE_THRESHOLD_ENABLE_OVER_DISABLE_UNDER  5
#define FEATURE_THRESHOLD_ENABLE_UNDER_DISABLE_OVER  6

// Pose landmarks - 3D joint positioning coordinates
// Horizontal rotation tracking values across body segments
#define FEATURE_POSE_X                      0
#define FEATURE_POSE_Y                      1
#define FEATURE_POSE_Z                      2
#define FEATURE_POSE_W                      3

// Gaze tracking features - precise eye movement measurement
// Encodes both vector and head translation data for comprehensive gaze analysis
#define FEATURE_GAZE_VECTOR                  0
#define FEATURE_GAZE_VECTOR_PITCH           1
#define FEATURE_GAZE_VECTOR_YAW             2
#define FEATURE_GAZE_HEADTRANSLATION_X       3
#define FEATURE_GAZE_HEADTRANSLATION_Y       4
#define FEATURE_GAZE_HEADTRANSLATION_Z       5
#define FEATURE_GAZE_DIRECTION_CENTER_POINT_X 6
#define FEATURE_GAZE_DIRECTION_CENTER_POINT_Y 7
#define FEATURE_GAZE_DIRECTION_CENTER_POINT_Z 8
#define FEATURE_GAZE_DIRECTION_VECTOR_X     9
#define FEATURE_GAZE_DIRECTION_VECTOR_Y    10
#define FEATURE_GAZE_DIRECTION_VECTOR_Z    11

// Facial expression metrics - comprehensive animation system for emotional expression
// Supports both single and multi-point expression rendering across facial regions
#define FEATURE_EXPRESSION_SINGLE        0
#define FEATURE_EXPRESSION_VECTOR        1
#define FEATURE_EXPRESSION_ADD           2
#define FEATURE_EXPRESSION_SUBSTRACT     3
#define FEATURE_EXPRESSION_DISTANCE     4
#define FEATURE_EXPRESSION_AVG          5

// Body segmentation features - complete skeletal point and joint angle analysis
// Includes wrist, jaw, and forehead tracking for comprehensive body detection
#define BODY_CONFIDENCE                  0
#define BODY_2D_POSX                     1
#define BODY_2D_POSY                     2
#define BODY_2D_DISTANCE                 3
#define BODY_2D_ROT                      4
#define BODY_2D_DIFF_X                   5
#define BODY_2D_DIFF_Y                   6
#define BODY_2D_DIFF                     7
#define BODY_2D_POS                      8
#define BODY_3D_POSX                     9
#define BODY_3D_POSY                    10
#define BODY_3D_POSZ                    11
#define BODY_3D_DISTANCE                12
#define BODY_3D_DIFF_X                  13
#define BODY_3D_DIFF_Y                  14
#define BODY_3D_DIFF_Z                  15
#define BODY_3D_POS                     16
#define BODY_3D_DIFF                    17
#define BODY_ANGLE_X                    18
#define BODY_ANGLE_Y                    19
#define BODY_ANGLE_Z                    20
#define BODY_ANGLE                      21

// NVIDIA-specific body angles - creative naming for specific joint rotations
#define BODY_ANGLE_X             18
#define BODY_ANGLE_Y             19
#define BODY_ANGLE_Z             20
#define BODY_ANGLE               21

#endif // MOVE_FILTER_ACTIONS_H
