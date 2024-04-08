#ifndef DET_RK3588__RKNN_MODEL_HPP_
#define DET_RK3588__RKNN_MODEL_HPP_

#include <mutex>

#include "opencv2/core/core.hpp"
#include "rknn_api.h"

#define RK3588_CORE_NUM 3

namespace det_rk3588
{
int GetCoreNum();

static void DumpTensorAttr(rknn_tensor_attr * attr);

double GetUs(struct timeval t);

static unsigned char * LoadData(FILE * fp, size_t ofst, size_t sz);

static unsigned char * LoadModel(const char * filename, int * model_size);

static int SaveFloat(const char * filename, float * output, int element_size);

class RknnModel
{
public:
  RknnModel(const std::string & model_path);

  ~RknnModel();

  int Init(rknn_context * ctx_in, bool is_child);

  rknn_context * GetPctx();

  cv::Mat Infer(cv::Mat & original_img);

private:
  int ret_;
  std::mutex mutex_;
  std::string model_path_;
  unsigned char * model_data_;

  rknn_context ctx_;
  rknn_input_output_num io_num_;
  rknn_tensor_attr * input_attrs_;
  rknn_tensor_attr * output_attrs_;
  rknn_input inputs_[1];

  int channel_;
  int width_;
  int height_;
  int img_width_;
  int img_height_;

  float nms_threshold_;
  float box_conf_threshold_;
};

}  // namespace det_rk3588

#endif  // DET_RK3588__RKNN_MODEL_HPP_
