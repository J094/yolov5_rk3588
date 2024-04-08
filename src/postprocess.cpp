#include "postprocess.hpp"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <vector>

#define LABEL_NALE_TXT_PATH "./model/labels_list.txt"

namespace det_rk3588
{

static char * labels[OBJ_CLASS_NUM];

const int anchor0[6] = {10, 13, 16, 30, 33, 23};
const int anchor1[6] = {30, 61, 62, 45, 59, 119};
const int anchor2[6] = {116, 90, 156, 198, 373, 326};

inline static int Clamp(float val, int min, int max)
{
  return val > min ? (val < max ? val : max) : min;
}

char * ReadLine(FILE * fp, char * buffer, int * len)
{
  int ch;
  int i = 0;
  size_t buff_len = 0;

  buffer = (char *)malloc(buff_len + 1);
  if (!buffer) return NULL;  // Out of memory

  while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
    buff_len++;
    void * tmp = realloc(buffer, buff_len + 1);
    if (tmp == NULL) {
      free(buffer);
      return NULL;  // Out of memory
    }
    buffer = (char *)tmp;

    buffer[i] = (char)ch;
    i++;
  }
  buffer[i] = '\0';

  *len = buff_len;

  // Detect end
  if (ch == EOF && (i == 0 || ferror(fp))) {
    free(buffer);
    return NULL;
  }
  return buffer;
}

int ReadLines(const char * filename, char * lines[], int max_line)
{
  FILE * file = fopen(filename, "r");
  char * s;
  int i = 0;
  int n = 0;

  if (file == NULL) {
    printf("Open %s fail!\n", filename);
    return -1;
  }

  while ((s = ReadLine(file, s, &n)) != NULL) {
    lines[i++] = s;
    if (i >= max_line) break;
  }
  fclose(file);
  return i;
}

int LoadLabelName(const char * location_filename, char * label[])
{
  printf("loadLabelName %s\n", location_filename);
  ReadLines(location_filename, label, OBJ_CLASS_NUM);
  return 0;
}

static float CalculateOverlap(
  float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
  float ymax1)
{
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) +
            (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}

static int Nms(
  int valid_count, std::vector<float> & output_locations, std::vector<int> class_ids,
  std::vector<int> & order, int filter_id, float threshold)
{
  for (int i = 0; i < valid_count; ++i) {
    if (order[i] == -1 || class_ids[i] != filter_id) {
      continue;
    }
    int n = order[i];
    for (int j = i + 1; j < valid_count; ++j) {
      int m = order[j];
      if (m == -1 || class_ids[i] != filter_id) {
        continue;
      }
      float xmin0 = output_locations[n * 4 + 0];
      float ymin0 = output_locations[n * 4 + 1];
      float xmax0 = output_locations[n * 4 + 0] + output_locations[n * 4 + 2];
      float ymax0 = output_locations[n * 4 + 1] + output_locations[n * 4 + 3];

      float xmin1 = output_locations[m * 4 + 0];
      float ymin1 = output_locations[m * 4 + 1];
      float xmax1 = output_locations[m * 4 + 0] + output_locations[m * 4 + 2];
      float ymax1 = output_locations[m * 4 + 1] + output_locations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

      if (iou > threshold) {
        order[j] = -1;
      }
    }
  }
  return 0;
}

static int QuickSortIndiceInverse(
  std::vector<float> & input, int left, int right, std::vector<int> & indices)
{
  float key;
  int key_index;
  int low = left;
  int high = right;
  if (left < right) {
    key_index = indices[left];
    key = input[left];
    while (low < high) {
      while (low < high && input[high] <= key) {
        high--;
      }
      input[low] = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key) {
        low++;
      }
      input[high] = input[low];
      indices[high] = indices[low];
    }
    input[low] = key;
    indices[low] = key_index;
    QuickSortIndiceInverse(input, left, low - 1, indices);
    QuickSortIndiceInverse(input, low + 1, right, indices);
  }
  return low;
}

static float Sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float Unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t Clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static int8_t QntF32ToAffine(float f32, int32_t zp, float scale)
{
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)Clip(dst_val, -128, 127);
  return res;
}

static float DeqntAffineToF32(int8_t qnt, int32_t zp, float scale)
{
  return ((float)qnt - (float)zp) * scale;
}

static int Process(
  int8_t * input, int * anchor, int grid_h, int grid_w, int height, int width, int stride,
  std::vector<float> & boxes, std::vector<float> & obj_probs, std::vector<int> & class_id,
  float threshold, int32_t zp, float scale)
{
  int valid_count = 0;
  int grid_len = grid_h * grid_w;
  int8_t thres_i8 = QntF32ToAffine(threshold, zp, scale);
  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < grid_h; i++) {
      for (int j = 0; j < grid_w; j++) {
        int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= thres_i8) {
          int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
          int8_t * in_ptr = input + offset;
          float box_x = (DeqntAffineToF32(*in_ptr, zp, scale)) * 2.0 - 0.5;
          float box_y = (DeqntAffineToF32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
          float box_w = (DeqntAffineToF32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
          float box_h = (DeqntAffineToF32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
          box_x = (box_x + j) * (float)stride;
          box_y = (box_y + i) * (float)stride;
          box_w = box_w * box_w * (float)anchor[a * 2];
          box_h = box_h * box_h * (float)anchor[a * 2 + 1];
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          int8_t max_class_probs = in_ptr[5 * grid_len];
          int maxClassId = 0;
          for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
            int8_t prob = in_ptr[(5 + k) * grid_len];
            if (prob > max_class_probs) {
              maxClassId = k;
              max_class_probs = prob;
            }
          }
          if (max_class_probs > thres_i8) {
            obj_probs.push_back(
              (DeqntAffineToF32(max_class_probs, zp, scale)) *
              (DeqntAffineToF32(box_confidence, zp, scale)));
            class_id.push_back(maxClassId);
            valid_count++;
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
          }
        }
      }
    }
  }
  return valid_count;
}

int PostProcess(
  int8_t * input0, int8_t * input1, int8_t * input2, int model_in_h, int model_in_w,
  float conf_threshold, float nms_threshold, BoxRect pads, float scale_w, float scale_h,
  std::vector<int32_t> & qnt_zps, std::vector<float> & qnt_scales, DetectResultGroup * group)
{
  static int init = -1;
  if (init == -1) {
    int ret = 0;
    ret = LoadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0) {
      return -1;
    }

    init = 0;
  }
  memset(group, 0, sizeof(DetectResultGroup));

  std::vector<float> filter_boxes;
  std::vector<float> obj_probs;
  std::vector<int> class_id;

  // stride 8
  int stride0 = 8;
  int grid_h0 = model_in_h / stride0;
  int grid_w0 = model_in_w / stride0;
  int valid_count0 = 0;
  valid_count0 = Process(
    input0, (int *)anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filter_boxes,
    obj_probs, class_id, conf_threshold, qnt_zps[0], qnt_scales[0]);

  // stride 16
  int stride1 = 16;
  int grid_h1 = model_in_h / stride1;
  int grid_w1 = model_in_w / stride1;
  int valid_count1 = 0;
  valid_count1 = Process(
    input1, (int *)anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filter_boxes,
    obj_probs, class_id, conf_threshold, qnt_zps[1], qnt_scales[1]);

  // stride 32
  int stride2 = 32;
  int grid_h2 = model_in_h / stride2;
  int grid_w2 = model_in_w / stride2;
  int valid_count2 = 0;
  valid_count2 = Process(
    input2, (int *)anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filter_boxes,
    obj_probs, class_id, conf_threshold, qnt_zps[2], qnt_scales[2]);

  int valid_count = valid_count0 + valid_count1 + valid_count2;
  // no object detect
  if (valid_count <= 0) {
    return 0;
  }

  std::vector<int> index_array;
  for (int i = 0; i < valid_count; ++i) {
    index_array.push_back(i);
  }

  QuickSortIndiceInverse(obj_probs, 0, valid_count - 1, index_array);

  std::set<int> class_set(std::begin(class_id), std::end(class_id));

  for (auto c : class_set) {
    Nms(valid_count, filter_boxes, class_id, index_array, c, nms_threshold);
  }

  int last_count = 0;
  group->count = 0;
  /* box valid detect target */
  for (int i = 0; i < valid_count; ++i) {
    if (index_array[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = index_array[i];

    float x1 = filter_boxes[n * 4 + 0] - pads.left;
    float y1 = filter_boxes[n * 4 + 1] - pads.top;
    float x2 = x1 + filter_boxes[n * 4 + 2];
    float y2 = y1 + filter_boxes[n * 4 + 3];
    int id = class_id[n];
    float obj_conf = obj_probs[i];

    group->results[last_count].box.left = (int)(Clamp(x1, 0, model_in_w) / scale_w);
    group->results[last_count].box.top = (int)(Clamp(y1, 0, model_in_h) / scale_h);
    group->results[last_count].box.right = (int)(Clamp(x2, 0, model_in_w) / scale_w);
    group->results[last_count].box.bottom = (int)(Clamp(y2, 0, model_in_h) / scale_h);
    group->results[last_count].prop = obj_conf;
    char * label = labels[id];
    strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

    // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
    // group->results[last_count].box.top,
    //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
    last_count++;
  }
  group->count = last_count;

  return 0;
}

void DeinitPostProcess()
{
  for (int i = 0; i < OBJ_CLASS_NUM; i++) {
    if (labels[i] != nullptr) {
      free(labels[i]);
      labels[i] = nullptr;
    }
  }
}

}  // namespace det_rk3588
