#include "preprocess.hpp"

namespace det_rk3588
{

void LetterBox(
  const cv::Mat & image, cv::Mat & padded_image, BoxRect & pads, const float scale,
  const cv::Size & target_size, const cv::Scalar & pad_color)
{
  // 调整图像大小
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(), scale, scale);

  // 计算填充大小
  int pad_width = target_size.width - resized_image.cols;
  int pad_height = target_size.height - resized_image.rows;

  pads.left = pad_width / 2;
  pads.right = pad_width - pads.left;
  pads.top = pad_height / 2;
  pads.bottom = pad_height - pads.top;

  // 在图像周围添加填充
  cv::copyMakeBorder(
    resized_image, padded_image, pads.top, pads.bottom, pads.left, pads.right, cv::BORDER_CONSTANT,
    pad_color);
}

int ResizeRga(
  rga_buffer_t & src, rga_buffer_t & dst, const cv::Mat & image, cv::Mat & resized_image,
  const cv::Size & target_size)
{
  im_rect src_rect;
  im_rect dst_rect;
  memset(&src_rect, 0, sizeof(src_rect));
  memset(&dst_rect, 0, sizeof(dst_rect));
  size_t img_width = image.cols;
  size_t img_height = image.rows;
  if (image.type() != CV_8UC3) {
    printf("source image type is %d!\n", image.type());
    return -1;
  }
  size_t target_width = target_size.width;
  size_t target_height = target_size.height;
  src = wrapbuffer_virtualaddr((void *)image.data, img_width, img_height, RK_FORMAT_RGB_888);
  dst = wrapbuffer_virtualaddr(
    (void *)resized_image.data, target_width, target_height, RK_FORMAT_RGB_888);
  int ret = imcheck(src, dst, src_rect, dst_rect);
  if (IM_STATUS_NOERROR != ret) {
    fprintf(stderr, "rga check error! %s", imStrError((IM_STATUS)ret));
    return -1;
  }
  IM_STATUS STATUS = imresize(src, dst);
  return 0;
}

}  // namespace det_rk3588
