#ifndef DET_RK3588__POSTPROCESS_HPP_
#define DET_RK3588__POSTPROCESS_HPP_

#include <stdint.h>

#include <vector>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 3
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

namespace det_rk3588
{

struct BoxRect
{
  int left;
  int right;
  int top;
  int bottom;
};

struct DetectResult
{
  char name[OBJ_NAME_MAX_SIZE];
  BoxRect box;
  float prop;
};

struct DetectResultGroup
{
  int id;
  int count;
  DetectResult results[OBJ_NUMB_MAX_SIZE];
};

int PostProcess(
  int8_t * input0, int8_t * input1, int8_t * input2, int model_in_h, int model_in_w,
  float conf_threshold, float nms_threshold, BoxRect pads, float scale_w, float scale_h,
  std::vector<int32_t> & qnt_zps, std::vector<float> & qnt_scales, DetectResultGroup * group);

void DeinitPostProcess();

}  // namespace det_rk3588

#endif  // DET_RK3588__POSTPROCESS_HPP_
