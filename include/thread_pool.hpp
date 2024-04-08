#ifndef DET_RK3588__THREAD_POOL_HPP_
#define DET_RK3588__THREAD_POOL_HPP_

#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

namespace det_rk3588
{

class ThreadPool
{
public:
  explicit ThreadPool(size_t max_threads)
  : quit_(false), current_threads_(0), idle_threads_(0), max_threads_(max_threads)
  {
  }

  ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}

  // disable copy operations
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool & operator=(const ThreadPool &) = delete;

  ~ThreadPool()
  {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      quit_ = true;
    }
    cv_.notify_all();

    for (auto & elem : threads_) {
      assert(elem.second.joinable());
      elem.second.join();
    }
  }

  template <typename F, typename... Ts>
  auto Submit(F && f, Ts &&... params) -> std::future<typename std::result_of<F(Ts...)>::type>
  {
    auto execute = std::bind(std::forward<F>(f), std::forward<Ts>(params)...);

    using ReturnType = typename std::result_of<F(Ts...)>::type;
    using PackagedTask = std::packaged_task<ReturnType()>;

    auto task = std::make_shared<PackagedTask>(std::move(execute));
    auto result = task->get_future();

    std::lock_guard<std::mutex> guard(mutex_);
    assert(!quit_);

    tasks_.emplace([task]() { (*task)(); });

    if (idle_threads_ > 0) {
      cv_.notify_one();
    } else if (current_threads_ < max_threads_) {
      std::thread t(&ThreadPool::Worker, this);
      assert(threads_.find(t.get_id()) == threads_.end());
      threads_[t.get_id()] = std::move(t);
      ++current_threads_;
    }

    return result;
  }

  size_t ThreadsNum()
  {
    std::lock_guard<std::mutex> guard(mutex_);
    return current_threads_;
  }

private:
  void Worker()
  {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ++idle_threads_;
        auto has_timed_out = !cv_.wait_for(
          lock, std::chrono::seconds(kWaitSeconds), [this]() { return quit_ || !tasks_.empty(); });
        --idle_threads_;
        if (tasks_.empty()) {
          if (quit_) {
            --current_threads_;
            return;
          }
          if (has_timed_out) {
            --current_threads_;
            JoinFinishedThreads();
            finished_thread_ids_.emplace(std::this_thread::get_id());
            return;
          }
        }
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  void JoinFinishedThreads()
  {
    while (!finished_thread_ids_.empty()) {
      auto id = std::move(finished_thread_ids_.front());
      finished_thread_ids_.pop();
      auto iter = threads_.find(id);

      assert(iter != threads_.end());
      assert(iter->second.joinable());

      iter->second.join();
      threads_.erase(iter);
    }
  }

private:
  static constexpr size_t kWaitSeconds = 2;

  bool quit_;
  size_t current_threads_;
  size_t idle_threads_;
  size_t max_threads_;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<std::function<void()>> tasks_;
  std::queue<std::thread::id> finished_thread_ids_;
  std::unordered_map<std::thread::id, std::thread> threads_;
};

constexpr size_t ThreadPool::kWaitSeconds;

}  // namespace det_rk3588

#endif  // DET_RK3588__THREAD_POOL_HPP_
