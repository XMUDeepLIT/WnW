#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

inline int32_t* i32(torch::Tensor& t) { return t.data_ptr<int32_t>(); }
inline const int32_t* ci32(const torch::Tensor& t) { return t.data_ptr<int32_t>(); }

}

class Planner {
 public:
  Planner(int num_layers, int num_kv_heads, int batch_size, int  )
      : L_(num_layers), H_(num_kv_heads), B_(batch_size), num_segs_(batch_size * num_kv_heads) {
    loaded_.resize(B_);
    counters_.resize(B_);
    c2s_.resize(B_);
    class_b_.resize(L_);
  }

  void set_config(int top_k_chunks, int eviction_lag) {
    top_k_ = top_k_chunks;
    evict_lag_ = eviction_lag;
  }

  void set_seg_count(std::vector<int> seg_count) { seg_count_ = std::move(seg_count); }

  void set_chunk_to_segs(int b, torch::Tensor flat_segs, torch::Tensor offsets) {
    const int32_t* fs = ci32(flat_segs);
    const int32_t* off = ci32(offsets);
    int num_chunks = static_cast<int>(offsets.numel()) - 1;
    if (static_cast<int>(c2s_.size()) < B_) c2s_.resize(B_);
    c2s_[b].assign(num_chunks, {});
    for (int c = 0; c < num_chunks; ++c) {
      c2s_[b][c].assign(fs + off[c], fs + off[c + 1]);
    }
  }

  void set_class_b(int layer, torch::Tensor kv_heads) {
    if (static_cast<int>(class_b_.size()) < L_) class_b_.resize(L_);
    const int32_t* p = ci32(kv_heads);
    class_b_[layer].assign(p, p + kv_heads.numel());
  }

  void set_restore_meta(torch::Tensor cpu_s, torch::Tensor cpu_e,
                        torch::Tensor gpu_s, torch::Tensor gpu_e) {
    const int32_t* cs = ci32(cpu_s);
    const int32_t* ce = ci32(cpu_e);
    const int32_t* gs = ci32(gpu_s);
    const int32_t* ge = ci32(gpu_e);
    cpu_s_.assign(L_, std::vector<std::vector<int>>(num_segs_));
    cpu_e_.assign(L_, std::vector<std::vector<int>>(num_segs_));
    gpu_s_.assign(L_, std::vector<std::vector<int>>(num_segs_));
    gpu_e_.assign(L_, std::vector<std::vector<int>>(num_segs_));
    long idx = 0;
    for (int l = 0; l < L_; ++l) {
      for (int fi = 0; fi < num_segs_; ++fi) {
        int sc = seg_count_[fi / H_];
        cpu_s_[l][fi].resize(sc);
        cpu_e_[l][fi].resize(sc);
        gpu_s_[l][fi].resize(sc);
        gpu_e_[l][fi].resize(sc);
        for (int s = 0; s < sc; ++s) {
          cpu_s_[l][fi][s] = cs[idx];
          cpu_e_[l][fi][s] = ce[idx];
          gpu_s_[l][fi][s] = gs[idx];
          gpu_e_[l][fi][s] = ge[idx];
          ++idx;
        }
      }
    }
  }

  void set_loaded(int b, torch::Tensor segs) {
    if (static_cast<int>(loaded_.size()) < B_) loaded_.resize(B_);
    const int32_t* p = ci32(segs);
    loaded_[b].clear();
    for (int64_t i = 0; i < segs.numel(); ++i) loaded_[b].insert(p[i]);
  }

  void set_seq_lens(torch::Tensor seq_lens_flat) {
    const int32_t* p = ci32(seq_lens_flat);
    seq_lens_.assign(L_, std::vector<int>(num_segs_));
    long idx = 0;
    for (int l = 0; l < L_; ++l)
      for (int fi = 0; fi < num_segs_; ++fi) seq_lens_[l][fi] = p[idx++];
    counters_.assign(B_, {});
    for (int b = 0; b < B_; ++b) counters_[b].assign(seg_count_[b], 0);
    prev_changed_ = false;
  }

  std::vector<int> get_loaded_segments(int b) {
    return std::vector<int>(loaded_[b].begin(), loaded_[b].end());
  }

  void set_restore_buffers(torch::Tensor cpu_k, torch::Tensor cpu_v,
                           torch::Tensor stg_k, torch::Tensor stg_v) {
    cpu_k_ = static_cast<char*>(cpu_k.data_ptr());
    cpu_v_ = static_cast<char*>(cpu_v.data_ptr());
    stg_k_ = static_cast<char*>(stg_k.data_ptr());
    stg_v_ = static_cast<char*>(stg_v.data_ptr());
    row_bytes_ = static_cast<long>(cpu_k.size(1)) * cpu_k.element_size();
  }

  void gather_restore() {
    long dst = 0;
    for (size_t i = 0; i + 1 < cpu_ranges_.size(); i += 2) {
      long s = cpu_ranges_[i];
      long n = static_cast<long>(cpu_ranges_[i + 1]) - s;
      std::memcpy(stg_k_ + dst * row_bytes_, cpu_k_ + s * row_bytes_, n * row_bytes_);
      std::memcpy(stg_v_ + dst * row_bytes_, cpu_v_ + s * row_bytes_, n * row_bytes_);
      dst += n;
    }
  }

  py::tuple compute_and_prepare(
      torch::Tensor topk_idx, torch::Tensor topk_off,
      torch::Tensor load_starts_2d, torch::Tensor load_lens_2d,
      torch::Tensor cu_new_2d, torch::Tensor new_lens_2d,
      torch::Tensor evict_offsets_2d, torch::Tensor evict_max,
      torch::Tensor evict_layer_ptr) {
    if (!prev_changed_) {
      for (int l = 0; l < L_; ++l)
        for (int fi = 0; fi < num_segs_; ++fi) seq_lens_[l][fi] += 1;
    }

    std::vector<std::set<int>> to_load(B_), to_evict(B_);
    const int32_t* tk = ci32(topk_idx);
    const int32_t* toff = ci32(topk_off);
    bool any_change = false;

    for (int b = 0; b < B_; ++b) {
      int lo = toff[b], hi = toff[b + 1];
      if (hi == lo) continue;

      std::set<int> needed;
      for (int j = lo; j < hi; ++j) {
        int ci = tk[j];
        for (int si : c2s_[b][ci]) needed.insert(si);
      }

      std::set<int>& loaded = loaded_[b];
      std::vector<int>& cnt = counters_[b];
      std::set<int> tl, te;
      for (int si : needed)
        if (loaded.find(si) == loaded.end()) tl.insert(si);
      for (int si : loaded) {
        if (needed.find(si) == needed.end()) {
          cnt[si] += 1;
          if (cnt[si] >= evict_lag_) te.insert(si);
        } else {
          cnt[si] = 0;
        }
      }
      for (int si : tl) loaded.insert(si);
      for (int si : te) {
        loaded.erase(si);
        cnt[si] = 0;
      }
      to_load[b] = std::move(tl);
      to_evict[b] = std::move(te);
      if (!to_load[b].empty() || !to_evict[b].empty()) any_change = true;
    }

    evict_starts_.clear();
    evict_ends_.clear();
    cpu_ranges_.clear();

    if (!any_change) {
      prev_changed_ = false;
      auto empty = torch::empty({0}, torch::kInt32);
      auto empty2 = torch::empty({0, 2}, torch::kInt32);
      return py::make_tuple(0, 0, empty, empty, empty2);
    }

    std::vector<int> changed_samples;
    for (int b = 0; b < B_; ++b)
      if (!to_load[b].empty() || !to_evict[b].empty()) changed_samples.push_back(b);

    std::vector<std::vector<int>> layer_new_lens(L_, std::vector<int>(num_segs_));
    std::vector<std::vector<std::array<int, 3>>> layer_load(L_);
    std::vector<std::vector<std::vector<std::pair<int, int>>>> layer_evict(
        L_, std::vector<std::vector<std::pair<int, int>>>(num_segs_));

    int global_transfer_start = 0;
    for (int l = 0; l < L_; ++l) {
      for (int fi = 0; fi < num_segs_; ++fi) layer_new_lens[l][fi] = seq_lens_[l][fi] + 1;

      for (int b : changed_samples) {
        for (int kv_h : class_b_[l]) {
          int fi = b * H_ + kv_h;
          int seg_len = seq_lens_[l][fi];
          int sc = seg_count_[b];
          std::vector<int>& gs = gpu_s_[l][fi];
          std::vector<int>& ge = gpu_e_[l][fi];
          const std::vector<int>& cs = cpu_s_[l][fi];
          const std::vector<int>& ce = cpu_e_[l][fi];

          int evicted = 0;
          if (!to_evict[b].empty()) {
            std::vector<std::pair<int, int>> ev;
            for (int seg_idx : to_evict[b]) {
              if (cs[seg_idx] >= 0 && gs[seg_idx] >= 0) {
                ev.emplace_back(gs[seg_idx], ge[seg_idx]);
                gs[seg_idx] = ge[seg_idx] = -1;
              }
            }
            std::sort(ev.begin(), ev.end());
            for (auto& pr : ev) evicted += pr.second - pr.first;

            if (!ev.empty()) {
              for (int si = 0; si < sc; ++si) {
                if (cs[si] < 0 || gs[si] < 0 || to_evict[b].count(si)) continue;
                int shift = 0;
                for (auto& pr : ev)
                  if (pr.second <= gs[si]) shift += pr.second - pr.first;
                gs[si] -= shift;
                ge[si] -= shift;
              }
            }
            layer_evict[l][fi] = std::move(ev);
          }

          int rel_cursor = seg_len - evicted;
          int loaded_n = 0;
          if (!to_load[b].empty()) {
            for (int seg_idx : to_load[b]) {
              if (cs[seg_idx] < 0) continue;
              int n = ce[seg_idx] - cs[seg_idx];
              cpu_ranges_.push_back(cs[seg_idx]);
              cpu_ranges_.push_back(ce[seg_idx]);
              layer_load[l].push_back({fi, global_transfer_start, n});
              gs[seg_idx] = rel_cursor;
              ge[seg_idx] = rel_cursor + n;
              rel_cursor += n;
              loaded_n += n;
              global_transfer_start += n;
            }
          }
          layer_new_lens[l][fi] = seg_len - evicted + loaded_n + 1;
        }
      }
    }

    int32_t* ls = i32(load_starts_2d);
    int32_t* ll = i32(load_lens_2d);
    int32_t* cun = i32(cu_new_2d);
    int32_t* nl = i32(new_lens_2d);
    int32_t* eoff = i32(evict_offsets_2d);
    int32_t* emax = i32(evict_max);
    int32_t* elp = i32(evict_layer_ptr);

    elp[0] = 0;
    for (int l = 0; l < L_; ++l) {
      int base = l * num_segs_;
      int cbase = l * (num_segs_ + 1);
      for (int fi = 0; fi < num_segs_; ++fi) {
        ls[base + fi] = -1;
        ll[base + fi] = 0;
      }
      for (auto& rec : layer_load[l]) {
        int fi = rec[0], transfer_start = rec[1], n = rec[2];
        if (ls[base + fi] < 0) ls[base + fi] = transfer_start;
        ll[base + fi] += n;
      }
      cun[cbase] = 0;
      for (int fi = 0; fi < num_segs_; ++fi) {
        nl[base + fi] = layer_new_lens[l][fi];
        cun[cbase + fi + 1] = cun[cbase + fi] + layer_new_lens[l][fi];
      }
      int emax_l = 0;
      eoff[cbase] = 0;
      for (int fi = 0; fi < num_segs_; ++fi) {
        auto& ranges = layer_evict[l][fi];
        for (auto& pr : ranges) {
          evict_starts_.push_back(pr.first);
          evict_ends_.push_back(pr.second);
        }
        eoff[cbase + fi + 1] = static_cast<int32_t>(evict_starts_.size()) -
                               (l == 0 ? 0 : elp[l]);
        emax_l = std::max(emax_l, static_cast<int>(ranges.size()));
      }
      emax[l] = emax_l;
      elp[l + 1] = static_cast<int32_t>(evict_starts_.size());
    }

    seq_lens_ = std::move(layer_new_lens);
    prev_changed_ = true;

    auto opt = torch::TensorOptions().dtype(torch::kInt32);
    int ne = static_cast<int>(evict_starts_.size());
    auto ev_s = torch::from_blob(evict_starts_.data(), {ne}, opt).clone();
    auto ev_e = torch::from_blob(evict_ends_.data(), {ne}, opt).clone();
    int nr = static_cast<int>(cpu_ranges_.size() / 2);
    auto ranges = torch::from_blob(cpu_ranges_.data(), {nr, 2}, opt).clone();

    return py::make_tuple(1, global_transfer_start, ev_s, ev_e, ranges);
  }

 private:
  int L_, H_, B_, num_segs_;
  int top_k_ = 0, evict_lag_ = 1;
  bool prev_changed_ = false;
  std::vector<int> seg_count_;
  std::vector<std::vector<std::vector<int>>> c2s_;
  std::vector<std::vector<int>> class_b_;
  std::vector<std::vector<std::vector<int>>> cpu_s_, cpu_e_, gpu_s_, gpu_e_;
  std::vector<std::set<int>> loaded_;
  std::vector<std::vector<int>> counters_;
  std::vector<std::vector<int>> seq_lens_;
  std::vector<int32_t> evict_starts_, evict_ends_, cpu_ranges_;
  char* cpu_k_ = nullptr;
  char* cpu_v_ = nullptr;
  char* stg_k_ = nullptr;
  char* stg_v_ = nullptr;
  long row_bytes_ = 0;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Planner>(m, "Planner")
      .def(py::init<int, int, int, int>())
      .def("set_config", &Planner::set_config)
      .def("set_seg_count", &Planner::set_seg_count)
      .def("set_chunk_to_segs", &Planner::set_chunk_to_segs)
      .def("set_class_b", &Planner::set_class_b)
      .def("set_restore_meta", &Planner::set_restore_meta)
      .def("set_loaded", &Planner::set_loaded)
      .def("set_seq_lens", &Planner::set_seq_lens)
      .def("get_loaded_segments", &Planner::get_loaded_segments)
      .def("set_restore_buffers", &Planner::set_restore_buffers)
      .def("gather_restore", &Planner::gather_restore)
      .def("compute_and_prepare", &Planner::compute_and_prepare);
}
