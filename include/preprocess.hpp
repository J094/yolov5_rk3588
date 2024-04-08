#ifndef DET_RK3588__PREPROCESS_HPP_
#define DET_RK3588__PREPROCESS_HPP_

#include <stdio.h>

#include <opencv2/opencv.hpp>

#include "im2d.h"
#include "postprocess.hpp"
#include "rga.h"

namespace det_rk3588
{

void LetterBox(
  const cv::Mat & image, cv::Mat & padded_image, BoxRect & pads, const float scale,
  const cv::Size & target_size, const cv::Scalar & pad_color = cv::Scalar(128, 128, 128));

int ResizeRga(
  rga_buffer_t & src, rga_buffer_t & dst, const cv::Mat & image, cv::Mat & resized_image,
  const cv::Size & target_size);

}  // namespace det_rk3588

#endif  // DET_RK3588__PREPROCESS_HPP_
