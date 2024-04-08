#ifndef DET_RK3588__RKNN_POOL_HPP_
#define DET_RK3588__RKNN_POOL_HPP_

#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "thread_pool.hpp"

namespace det_rk3588
{

template <typename ModelType, typename InputType, typename OutputType>
class RknnPool
{
public:
  RknnPool(const std::string model_path, int thread_num);

  ~RknnPool();

  int Init();

  // model inference
  int Put(InputType & input_data);

  // get result
  int Get(OutputType & output_data);

protected:
  int GetModelId();

private:
  int thread_num_;
  std::string model_path_;

  long long id_;

  std::mutex id_mutex_;
  std::mutex queue_mutex_;

  std::unique_ptr<ThreadPool> thread_pool_;
  std::queue<std::future<OutputType>> futures_;
  std::vector<std::shared_ptr<ModelType>> models_;
};

template <typename ModelType, typename InputType, typename OutputType>
RknnPool<ModelType, InputType, OutputType>::RknnPool(const std::string model_path, int thread_num)
{
  model_path_ = model_path;
  thread_num_ = thread_num;
  id_ = 0;
}

template <typename ModelType, typename InputType, typename OutputType>
int RknnPool<ModelType, InputType, OutputType>::Init()
{
  try {
    thread_pool_ = std::make_unique<ThreadPool>(thread_num_);
    for (int i = 0; i < thread_num_; i++)
      models_.push_back(std::make_shared<ModelType>(model_path_.c_str()));
  } catch (const std::bad_alloc & e) {
    std::cout << "Out of memory: " << e.what() << std::endl;
    return -1;
  }
  // initialize rknn model
  for (int i = 0, ret = 0; i < thread_num_; i++) {
    // all models share weights with the first one
    ret = models_[i]->Init(models_[0]->GetPctx(), i != 0);
    if (ret != 0) return ret;
  }

  return 0;
}

template <typename ModelType, typename InputType, typename OutputType>
int RknnPool<ModelType, InputType, OutputType>::GetModelId()
{
  std::lock_guard<std::mutex> lock(id_mutex_);
  int model_id = id_ % thread_num_;
  id_++;
  return model_id;
}

template <typename ModelType, typename InputType, typename OutputType>
int RknnPool<ModelType, InputType, OutputType>::Put(InputType & input_data)
{
  futures_.push(thread_pool_->Submit(&ModelType::Infer, models_[GetModelId()], input_data));
  return 0;
}

template <typename ModelType, typename InputType, typename OutputType>
int RknnPool<ModelType, InputType, OutputType>::Get(OutputType & output_data)
{
  std::lock_guard<std::mutex> lock(queue_mutex_);
  if (futures_.empty() == true) return 1;
  output_data = futures_.front().get();
  futures_.pop();
  return 0;
}

template <typename ModelType, typename InputType, typename OutputType>
RknnPool<ModelType, InputType, OutputType>::~RknnPool()
{
  while (!futures_.empty()) {
    OutputType tmp = futures_.front().get();
    futures_.pop();
  }
}

}  // namespace det_rk3588

#endif  // DET_RK3588__RKNN_POOL_HPP_