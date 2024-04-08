#include "rknn_model.hpp"

#include "postprocess.hpp"
#include "preprocess.hpp"

namespace det_rk3588
{

int GetCoreNum()
{
  static int core_num = 0;
  static std::mutex mtx;

  std::lock_guard<std::mutex> lock(mtx);

  int tmp = core_num & RK3588_CORE_NUM;
  core_num++;
  return tmp;
}

static void DumpTensorAttr(rknn_tensor_attr * attr)
{
  std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
  for (int i = 1; i < attr->n_dims; ++i) {
    shape_str += ", " + std::to_string(attr->dims[i]);
  }

  //  printf(
  //    "  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, "
  //    "size_with_stride=%d, fmt=%s, "
  //    "type=%s, qnt_type=%s, "
  //    "zp=%d, scale=%f\n",
  //    attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size,
  //    attr->w_stride, attr->size_with_stride, get_format_string(attr->fmt),
  //    get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double GetUs(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char * LoadData(FILE * fp, size_t ofst, size_t sz)
{
  unsigned char * data;
  int ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char *)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char * LoadModel(const char * filename, int * model_size)
{
  FILE * fp;
  unsigned char * data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = LoadData(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int SaveFloat(const char * filename, float * output, int element_size)
{
  FILE * fp;
  fp = fopen(filename, "w");
  for (int i = 0; i < element_size; i++) {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}

RknnModel::RknnModel(const std::string & model_path)
{
  model_path_ = model_path;
  nms_threshold_ = NMS_THRESH;
  box_conf_threshold_ = BOX_THRESH;
}

int RknnModel::Init(rknn_context * ctx_in, bool share_weight)
{
  printf("Loading model...\n");
  // model weights reusable
  if (share_weight == true) {
    ret_ = rknn_dup_context(ctx_in, &ctx_);
  } else {
    int model_data_size = 0;
    model_data_ = LoadModel(model_path_.c_str(), &model_data_size);
    ret_ = rknn_init(&ctx_, model_data_, model_data_size, 0, NULL);
  }
  if (ret_ < 0) {
    printf("rknn init error. ret=%d\n", ret_);
    return -1;
  }

  // set npu core for this model
  rknn_core_mask core_mask;
  switch (GetCoreNum()) {
    case 0:
      core_mask = RKNN_NPU_CORE_0;
      break;
    case 1:
      core_mask = RKNN_NPU_CORE_1;
      break;
    case 2:
      core_mask = RKNN_NPU_CORE_2;
      break;
  }
  ret_ = rknn_set_core_mask(ctx_, core_mask);
  if (ret_ < 0) {
    printf("rknn set core mask error. ret=%d\n", ret_);
    return -1;
  }

  rknn_sdk_version version;
  ret_ = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  if (ret_ < 0) {
    printf("rknn query sdk version error. ret=%d\n", ret_);
    return -1;
  }
  printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

  ret_ = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret_ < 0) {
    printf("rknn query in out num error. ret=%d\n", ret_);
    return -1;
  }
  printf("model input num: %d, output num: %d\n", io_num_.n_input, io_num_.n_output);

  input_attrs_ = (rknn_tensor_attr *)calloc(io_num_.n_input, sizeof(rknn_tensor_attr));
  for (int i = 0; i < io_num_.n_input; i++) {
    input_attrs_[i].index = i;
    ret_ = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]), sizeof(rknn_tensor_attr));
    if (ret_ < 0) {
      printf("rknn query input attr error. ret=%d\n", ret_);
      return -1;
    }
    DumpTensorAttr(&(input_attrs_[i]));
  }

  output_attrs_ = (rknn_tensor_attr *)calloc(io_num_.n_output, sizeof(rknn_tensor_attr));
  for (int i = 0; i < io_num_.n_output; i++) {
    output_attrs_[i].index = i;
    ret_ = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr));
    if (ret_ < 0) {
      printf("rknn query output attr error. ret=%d\n", ret_);
      return -1;
    }
    DumpTensorAttr(&(output_attrs_[i]));
  }

  if (input_attrs_[0].fmt == RKNN_TENSOR_NCHW) {
    printf("model is NCHW input fmt\n");
    channel_ = input_attrs_[0].dims[1];
    height_ = input_attrs_[0].dims[2];
    width_ = input_attrs_[0].dims[3];
  } else {
    printf("model is NHWC input fmt\n");
    height_ = input_attrs_[0].dims[1];
    width_ = input_attrs_[0].dims[2];
    channel_ = input_attrs_[0].dims[3];
  }
  printf("model input height=%d, width=%d, channel=%d\n", height_, width_, channel_);

  memset(inputs_, 0, sizeof(inputs_));
  inputs_[0].index = 0;
  inputs_[0].type = RKNN_TENSOR_UINT8;
  inputs_[0].size = width_ * height_ * channel_;
  inputs_[0].fmt = RKNN_TENSOR_NHWC;
  inputs_[0].pass_through = 0;

  return 0;
}

rknn_context * RknnModel::GetPctx() { return &ctx_; }

cv::Mat RknnModel::Infer(cv::Mat & original_img)
{
  std::lock_guard<std::mutex> lock(mutex_);
  cv::Mat img;
  cv::cvtColor(original_img, img, cv::COLOR_BGR2RGB);
  img_width_ = img.cols;
  img_height_ = img.rows;

  BoxRect pads;
  memset(&pads, 0, sizeof(BoxRect));
  cv::Size target_size(width_, height_);
  cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);
  // calculate scaling ratio
  float scale_w = (float)target_size.width / img.cols;
  float scale_h = (float)target_size.height / img.rows;

  // image scaling
  if (img_width_ != width_ || img_height_ != height_) {
    float min_scale = std::min(scale_w, scale_h);
    scale_w = min_scale;
    scale_h = min_scale;
    LetterBox(img, resized_img, pads, min_scale, target_size);
    inputs_[0].buf = resized_img.data;
  } else {
    inputs_[0].buf = img.data;
  }

  rknn_inputs_set(ctx_, io_num_.n_input, inputs_);

  rknn_output outputs[io_num_.n_output];
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < io_num_.n_output; i++) {
    outputs[i].want_float = 0;
  }

  // model inference
  ret_ = rknn_run(ctx_, NULL);
  ret_ = rknn_outputs_get(ctx_, io_num_.n_output, outputs, NULL);

  // postprocessing
  DetectResultGroup detect_result_group;
  std::vector<float> out_scales;
  std::vector<int32_t> out_zps;
  for (int i = 0; i < io_num_.n_output; ++i) {
    out_scales.push_back(output_attrs_[i].scale);
    out_zps.push_back(output_attrs_[i].zp);
  }
  PostProcess(
    (int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height_, width_,
    box_conf_threshold_, nms_threshold_, pads, scale_w, scale_h, out_zps, out_scales,
    &detect_result_group);

  // draw box
  char text[256];
  for (int i = 0; i < detect_result_group.count; i++) {
    DetectResult * det_result = &(detect_result_group.results[i]);
    sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
    // print information about the predicted object
    printf(
      "%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
      det_result->box.right, det_result->box.bottom, det_result->prop);
    int x1 = det_result->box.left;
    int y1 = det_result->box.top;
    int x2 = det_result->box.right;
    int y2 = det_result->box.bottom;
    rectangle(original_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(256, 0, 0, 256), 3);
    putText(
      original_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4,
      cv::Scalar(255, 255, 255));
  }

  ret_ = rknn_outputs_release(ctx_, io_num_.n_output, outputs);

  return original_img;
}

RknnModel::~RknnModel()
{
  DeinitPostProcess();

  ret_ = rknn_destroy(ctx_);

  if (model_data_) {
    free(model_data_);
  }
  if (input_attrs_) {
    free(input_attrs_);
  }
  if (output_attrs_) {
    free(output_attrs_);
  }
}

}  // namespace det_rk3588
