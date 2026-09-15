#include "serving/kv_cache.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace serving {
namespace {

class PagedKvCache final : public KvCache {
public:
  PagedKvCache(ModelConfig model, bool prefix_cache_enabled)
      : model_(std::move(model)), prefix_cache_enabled_(prefix_cache_enabled),
        blocks_(model_.total_blocks) {
    free_block_ids_.reserve(blocks_.size());
    for (std::size_t index = blocks_.size(); index > 0; --index) {
      free_block_ids_.push_back(static_cast<int>(index - 1));
    }
  }

  AdmissionResult admit(RequestRuntime &request) override {
    const std::size_t block_count =
        ceil_div(request.prompt_tokens(), model_.block_tokens);
    if (block_count > blocks_.size()) {
      return AdmissionResult::Impossible;
    }

    std::vector<int> cached_ids;
    std::size_t cached_tokens = 0;
    if (prefix_cache_enabled_) {
      const std::size_t complete_prompt_blocks =
          request.prompt_tokens() / model_.block_tokens;
      cached_ids.reserve(complete_prompt_blocks);
      for (std::size_t block_index = 0; block_index < complete_prompt_blocks;
           ++block_index) {
        const auto begin =
            request.spec.prompt_tokens.begin() +
            static_cast<std::ptrdiff_t>(block_index * model_.block_tokens);
        const auto end =
            begin + static_cast<std::ptrdiff_t>(model_.block_tokens);
        const std::uint64_t hash = hash_prompt_block(std::span(begin, end));
        const auto cache_it = cache_index_.find(hash);
        if (cache_it == cache_index_.end()) {
          break;
        }
        const Block &block =
            blocks_[static_cast<std::size_t>(cache_it->second)];
        if (!block.cache_pinned) {
          break;
        }
        cached_ids.push_back(cache_it->second);
        cached_tokens += model_.block_tokens;
      }
    }

    const std::size_t new_blocks = block_count - cached_ids.size();
    if (new_blocks > blocks_.size()) {
      return AdmissionResult::Impossible;
    }
    if (!ensure_capacity(new_blocks)) {
      return AdmissionResult::TemporarilyUnavailable;
    }

    request.block_table.assign(block_count, -1);
    request.prefix_hit_tokens = cached_tokens;
    request.prefetched_tokens = cached_tokens;
    request.stored_tokens = cached_tokens;
    request.reserved_blocks = new_blocks;
    reserved_unmaterialized_ += new_blocks;

    for (std::size_t index = 0; index < cached_ids.size(); ++index) {
      const int block_id = cached_ids[index];
      request.block_table[index] = block_id;
      Block &block = blocks_[static_cast<std::size_t>(block_id)];
      ++block.request_refs;
      block.last_used = ++clock_;
    }
    return AdmissionResult::Admitted;
  }

  bool append_tokens(RequestRuntime &request,
                     const std::size_t token_count) override {
    if (token_count > request.total_tokens() - request.stored_tokens) {
      return false;
    }
    const std::size_t needed_blocks =
        ceil_div(request.stored_tokens + token_count, model_.block_tokens);
    if (needed_blocks > blocks_.size()) {
      return false;
    }
    const std::size_t growth_blocks =
        needed_blocks > request.block_table.size()
            ? needed_blocks - request.block_table.size()
            : 0;
    // Prompt capacity is reserved at admission. Decode growth is not:
    // pages are obtained only when the sequence crosses a block boundary.
    if (!ensure_capacity(growth_blocks)) {
      return false;
    }
    if (growth_blocks > 0) {
      request.block_table.resize(needed_blocks, -1);
    }
    for (std::size_t token = 0; token < token_count; ++token) {
      const std::size_t block_index =
          request.stored_tokens / model_.block_tokens;
      int &block_id = request.block_table[block_index];
      if (block_id < 0) {
        block_id = free_block_ids_.back();
        free_block_ids_.pop_back();
        if (request.reserved_blocks > 0) {
          --request.reserved_blocks;
          --reserved_unmaterialized_;
        }
        blocks_[static_cast<std::size_t>(block_id)].request_refs = 1;
      }

      Block &block = blocks_[static_cast<std::size_t>(block_id)];
      const std::size_t offset = request.stored_tokens % model_.block_tokens;
      block.used_tokens = std::max(block.used_tokens, offset + 1);
      block.last_used = ++clock_;
      ++request.stored_tokens;
    }
    return true;
  }

  void cache_completed_prompt_block(RequestRuntime &request,
                                    const std::size_t block_index,
                                    const std::uint64_t block_hash) override {
    if (!prefix_cache_enabled_ || block_index >= request.block_table.size()) {
      return;
    }
    const int block_id = request.block_table[block_index];
    if (block_id < 0) {
      return;
    }
    const auto existing = cache_index_.find(block_hash);
    if (existing != cache_index_.end()) {
      blocks_[static_cast<std::size_t>(existing->second)].last_used = ++clock_;
      return;
    }
    Block &block = blocks_[static_cast<std::size_t>(block_id)];
    block.cache_pinned = true;
    block.last_used = ++clock_;
    cache_index_.emplace(block_hash, block_id);
  }

  void release(RequestRuntime &request) override {
    if (request.reserved_blocks > 0) {
      reserved_unmaterialized_ -= request.reserved_blocks;
      request.reserved_blocks = 0;
    }
    for (const int block_id : request.block_table) {
      if (block_id < 0) {
        continue;
      }
      Block &block = blocks_[static_cast<std::size_t>(block_id)];
      if (block.request_refs > 0) {
        --block.request_refs;
      }
      if (block.request_refs == 0 && !block.cache_pinned) {
        block.used_tokens = 0;
        free_block_ids_.push_back(block_id);
      }
    }
    request.block_table.clear();
  }

  [[nodiscard]] KvCacheStats stats() const override {
    KvCacheStats result;
    result.total_blocks = blocks_.size();
    result.available_blocks =
        free_block_ids_.size() >= reserved_unmaterialized_
            ? free_block_ids_.size() - reserved_unmaterialized_
            : 0;
    result.largest_free_run_blocks = result.available_blocks;
    for (const Block &block : blocks_) {
      if (block.request_refs > 0 || block.cache_pinned) {
        ++result.allocated_blocks;
      }
      if (block.request_refs > 0) {
        ++result.active_blocks;
      }
      if (block.cache_pinned) {
        ++result.cached_blocks;
      }
      if (block.request_refs > 0 || block.cache_pinned) {
        result.physical_block_utilization +=
            static_cast<double>(block.used_tokens);
      }
    }
    const double capacity =
        static_cast<double>(result.allocated_blocks * model_.block_tokens);
    if (capacity > 0.0) {
      result.physical_block_utilization /= capacity;
      result.internal_waste_ratio = 1.0 - result.physical_block_utilization;
    }
    result.cache_evictions = cache_evictions_;
    return result;
  }

private:
  struct Block {
    std::size_t used_tokens = 0;
    std::size_t request_refs = 0;
    std::uint64_t last_used = 0;
    bool cache_pinned = false;
  };

  static std::size_t ceil_div(const std::size_t value,
                              const std::size_t divisor) {
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
  }

  bool ensure_capacity(const std::size_t required_blocks) {
    while (free_block_ids_.size() <
           reserved_unmaterialized_ + required_blocks) {
      if (!evict_one_cache_entry()) {
        return false;
      }
    }
    return true;
  }

  bool evict_one_cache_entry() {
    auto candidate = cache_index_.end();
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (auto it = cache_index_.begin(); it != cache_index_.end(); ++it) {
      const Block &block = blocks_[static_cast<std::size_t>(it->second)];
      if (block.request_refs == 0 && block.last_used <= oldest) {
        candidate = it;
        oldest = block.last_used;
      }
    }
    if (candidate == cache_index_.end()) {
      return false;
    }

    const int block_id = candidate->second;
    blocks_[static_cast<std::size_t>(block_id)].cache_pinned = false;
    blocks_[static_cast<std::size_t>(block_id)].used_tokens = 0;
    cache_index_.erase(candidate);
    free_block_ids_.push_back(block_id);
    ++cache_evictions_;
    return true;
  }

  ModelConfig model_;
  bool prefix_cache_enabled_ = false;
  std::vector<Block> blocks_;
  std::vector<int> free_block_ids_;
  std::unordered_map<std::uint64_t, int> cache_index_;
  std::uint64_t clock_ = 0;
  std::size_t reserved_unmaterialized_ = 0;
  std::size_t cache_evictions_ = 0;
};

class ContiguousKvCache final : public KvCache {
public:
  explicit ContiguousKvCache(ModelConfig model)
      : model_(std::move(model)), blocks_(model_.total_blocks) {}

  AdmissionResult admit(RequestRuntime &request) override {
    const std::size_t needed =
        ceil_div(request.total_tokens(), model_.block_tokens);
    if (needed > blocks_.size()) {
      return AdmissionResult::Impossible;
    }

    std::size_t run_start = 0;
    std::size_t run_length = 0;
    for (std::size_t index = 0; index < blocks_.size(); ++index) {
      if (!blocks_[index].owned) {
        if (run_length == 0) {
          run_start = index;
        }
        ++run_length;
        if (run_length == needed) {
          request.block_table.clear();
          for (std::size_t block = run_start; block < run_start + needed;
               ++block) {
            blocks_[block].owned = true;
            request.block_table.push_back(static_cast<int>(block));
          }
          return AdmissionResult::Admitted;
        }
      } else {
        run_length = 0;
      }
    }
    return AdmissionResult::TemporarilyUnavailable;
  }

  bool append_tokens(RequestRuntime &request,
                     const std::size_t token_count) override {
    for (std::size_t token = 0; token < token_count; ++token) {
      if (request.stored_tokens >=
          request.block_table.size() * model_.block_tokens) {
        return false;
      }
      const std::size_t block_index =
          request.stored_tokens / model_.block_tokens;
      Block &block =
          blocks_[static_cast<std::size_t>(request.block_table[block_index])];
      block.used_tokens = std::max(
          block.used_tokens, request.stored_tokens % model_.block_tokens + 1);
      ++request.stored_tokens;
    }
    return true;
  }

  void cache_completed_prompt_block(RequestRuntime &, std::size_t,
                                    std::uint64_t) override {}

  void release(RequestRuntime &request) override {
    for (const int block_id : request.block_table) {
      if (block_id < 0) {
        continue;
      }
      Block &block = blocks_[static_cast<std::size_t>(block_id)];
      block = Block{};
    }
    request.block_table.clear();
  }

  [[nodiscard]] KvCacheStats stats() const override {
    KvCacheStats result;
    result.total_blocks = blocks_.size();
    std::size_t current_free_run = 0;
    for (const Block &block : blocks_) {
      if (block.owned) {
        ++result.allocated_blocks;
        ++result.active_blocks;
        result.physical_block_utilization +=
            static_cast<double>(block.used_tokens);
        current_free_run = 0;
      } else {
        ++current_free_run;
        result.largest_free_run_blocks =
            std::max(result.largest_free_run_blocks, current_free_run);
      }
    }
    result.available_blocks = result.total_blocks - result.allocated_blocks;
    const double capacity =
        static_cast<double>(result.allocated_blocks * model_.block_tokens);
    if (capacity > 0.0) {
      result.physical_block_utilization /= capacity;
      result.internal_waste_ratio = 1.0 - result.physical_block_utilization;
    }
    if (result.available_blocks > 0) {
      result.external_fragmentation_ratio =
          1.0 - static_cast<double>(result.largest_free_run_blocks) /
                    static_cast<double>(result.available_blocks);
    }
    return result;
  }

private:
  struct Block {
    bool owned = false;
    std::size_t used_tokens = 0;
  };

  static std::size_t ceil_div(const std::size_t value,
                              const std::size_t divisor) {
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
  }

  ModelConfig model_;
  std::vector<Block> blocks_;
};

} // namespace

std::unique_ptr<KvCache> make_kv_cache(const ModelConfig &model,
                                       const ServiceConfig &service) {
  if (service.kv_layout == KvLayout::Contiguous) {
    return std::make_unique<ContiguousKvCache>(model);
  }
  return std::make_unique<PagedKvCache>(model, service.prefix_cache_enabled);
}

std::uint64_t hash_prompt_block(const std::span<const std::uint32_t> tokens) {
  // FNV-1a is enough for deterministic simulator keys; a service would need a
  // stronger hash.
  std::uint64_t hash = 14'695'981'039'346'656'037ULL;
  for (const std::uint32_t token : tokens) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      hash ^= static_cast<std::uint64_t>((token >> shift) & 0xffU);
      hash *= 1'099'511'628'211ULL;
    }
  }
  return hash;
}

} // namespace serving
