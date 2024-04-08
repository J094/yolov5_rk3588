#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include <memory>
#include <opencv2/opencv.hpp>

#include "rknn_model.hpp"
#include "rknn_pool.hpp"

#define THREAD_NUM 6

using namespace det_rk3588;

int main(int argc, char ** argv)
{
  char * model_path = nullptr;
  char * video_path = nullptr;
  char * save_path = nullptr;
  if (argc != 4) {
    printf("Usage: %s <model path> <video path> <save path> \n", argv[0]);
    return -1;
  }
  model_path = (char *)argv[1];
  video_path = (char *)argv[2];
  save_path = (char *)argv[3];

  // initialize rknn thread pool
  RknnPool<RknnModel, cv::Mat, cv::Mat> rknn_pool(model_path, THREAD_NUM);
  if (rknn_pool.Init() != 0) {
    printf("rknn pool init failed.\n");
    return -1;
  }

  cv::VideoCapture video_capture;
  if (strlen(video_path) <= 2) {
    video_capture.open(22);
  } else {
    video_capture.open(video_path);
  }

  int frame_width = static_cast<int>(video_capture.get(3));
  int frame_height = static_cast<int>(video_capture.get(4));
  cv::Size frame_size(frame_width, frame_height);
  int fps = 10;
  cv::VideoWriter video_writer(
    save_path, cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, frame_size);

  timeval time;
  gettimeofday(&time, nullptr);
  auto start_time = GetUs(time);

  int frames = 0;
  auto before_time = start_time;
  while (video_capture.isOpened()) {
    cv::Mat img;
    if (video_capture.read(img) == false) {
      break;
    }
    if (rknn_pool.Put(img) != 0) {
      break;
    }

    if (frames >= THREAD_NUM && rknn_pool.Get(img) != 0) {
      break;
    }
    frames++;

    if (frames % 120 == 0) {
      gettimeofday(&time, nullptr);
      auto current_time = GetUs(time);
      printf("120 frames, average fps: %f\n", 120.0 / float(current_time - before_time) * 1e6);
      before_time = current_time;
    }

    usleep(1);

    video_writer.write(img);
  }

  while (true) {
    cv::Mat img;
    if (rknn_pool.Get(img) != 0) {
      break;
    }
    frames++;

    usleep(1);
  }

  gettimeofday(&time, nullptr);
  auto end_time = GetUs(time);
  printf("average fps: %f\n", float(frames) / float(end_time - start_time) * 1e6);

  return 0;
}
