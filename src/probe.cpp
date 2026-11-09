#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/cuda.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/optim/adamw.h>
#include <torch/serialize/input-archive.h>
#include <torch/serialize/output-archive.h>
#include <torch/types.h>
#include <torch/version.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {
using Tensor = torch::Tensor;
using Clock = std::chrono::steady_clock;
using Trace = std::vector<std::pair<std::string, Tensor>>;

struct Options {
  std::string mode = "verify", device = "cuda", output;
  int64_t width = 32, layers = 2, heads = 4, seq = 16, batch = 4, vocab = 32;
  int64_t steps = 40, warmup = 5, seed = 20261104, threads = 2;
  double seconds = 0;
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void help() {
  std::cout <<
      "Usage: rnn2_probe [verify|benchmark] [options]\n"
      "  --device cpu|cuda   Explicit device; CUDA is also LibTorch's ROCm name.\n"
      "  --output DIR        Required artifact directory (checkpoint/config).\n"
      "  --width N --layers N --heads N --seq N --batch N --vocab N\n"
      "  --steps N           Learning steps (verify) or measured steps (benchmark).\n"
      "  --warmup N          Untimed training steps before benchmark.\n"
      "  --seconds S         Optional benchmark time budget; stop after the first\n"
      "                      completed step reaching S, or --steps, whichever first.\n"
      "  --seed N --threads N\n"
      "Defaults: verify, cuda, width32/layers2/heads4/seq16/batch4/vocab32,\n"
      "steps40, warmup5, seconds0, seed20261104, CPU threads2.\n"
      "All operations and AdamW states use FP32. Explicit causal attention,\n"
      "4x GELU feed-forward, learned token/position embeddings, untied output.\n"
      "The disposable model starts from seeded random weights. This is a\n"
      "numerical/training probe, not a selected student or capability benchmark.\n"
      "verify on cuda compares CPU and GPU; verify on cpu is CPU-only coverage.\n";
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") { help(); std::exit(0); }
    if (arg == "verify" || arg == "benchmark") { o.mode = arg; continue; }
    require(i + 1 < argc, "Missing value for " + arg);
    const std::string value = argv[++i];
    if (arg == "--device") o.device = value;
    else if (arg == "--output") o.output = value;
    else if (arg == "--seconds") {
      size_t parsed = 0;
      o.seconds = std::stod(value, &parsed);
      require(parsed == value.size(), "Invalid number: " + value);
    }
    else {
      size_t parsed = 0;
      const auto n = std::stoll(value, &parsed);
      require(parsed == value.size(), "Invalid integer: " + value);
      if (arg == "--width") o.width = n;
      else if (arg == "--layers") o.layers = n;
      else if (arg == "--heads") o.heads = n;
      else if (arg == "--seq") o.seq = n;
      else if (arg == "--batch") o.batch = n;
      else if (arg == "--vocab") o.vocab = n;
      else if (arg == "--steps") o.steps = n;
      else if (arg == "--warmup") o.warmup = n;
      else if (arg == "--seed") o.seed = n;
      else if (arg == "--threads") o.threads = n;
      else throw std::runtime_error("Unknown option: " + arg);
    }
  }
  require(o.device == "cpu" || o.device == "cuda", "Device must be cpu or cuda");
  require(!o.output.empty(), "--output DIR is required");
  require(o.width > 0 && o.layers > 0 && o.heads > 0 && o.width % o.heads == 0,
          "Positive width/layers/heads and width divisible by heads required");
  require(o.seq >= 2 && o.batch > 0 && o.vocab >= 2 && o.steps > 0 &&
              o.warmup >= 0 && o.threads > 0 && std::isfinite(o.seconds) && o.seconds >= 0,
          "Invalid sequence/batch/vocab/steps/warmup/threads/seconds");
  return o;
}

void sync(const torch::Device& device) {
  if (device.is_cuda()) torch::cuda::synchronize(device.index());
}

void capture(Trace* trace, const std::string& name, const Tensor& value) {
  if (trace) trace->emplace_back(name, value.detach());
}

struct Block : torch::nn::Module {
  int64_t heads, head_width;
  torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
  torch::nn::Linear qkv{nullptr}, projection{nullptr}, up{nullptr}, down{nullptr};
  Block(int64_t width, int64_t nheads) : heads(nheads), head_width(width / nheads) {
    ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({width})));
    ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({width})));
    qkv = register_module("qkv", torch::nn::Linear(torch::nn::LinearOptions(width, 3 * width).bias(false)));
    projection = register_module("projection", torch::nn::Linear(width, width));
    up = register_module("up", torch::nn::Linear(width, 4 * width));
    down = register_module("down", torch::nn::Linear(4 * width, width));
  }
  Tensor forward(Tensor x, Trace* trace, const std::string& prefix) {
    const auto b = x.size(0), t = x.size(1), d = x.size(2);
    auto normalized = ln1->forward(x);
    capture(trace, prefix + ".ln1", normalized);
    auto parts = qkv->forward(normalized).chunk(3, -1);
    auto q = parts[0].view({b, t, heads, head_width}).transpose(1, 2);
    auto k = parts[1].view({b, t, heads, head_width}).transpose(1, 2);
    auto v = parts[2].view({b, t, heads, head_width}).transpose(1, 2);
    auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(double(head_width));
    auto mask = torch::ones({t, t}, x.options().dtype(torch::kBool)).triu(1);
    auto attention = torch::softmax(scores.masked_fill(mask, -std::numeric_limits<float>::infinity()), -1);
    capture(trace, prefix + ".attention", attention);
    auto context = torch::matmul(attention, v).transpose(1, 2).contiguous().view({b, t, d});
    x = x + projection->forward(context);
    auto hidden = torch::gelu(up->forward(ln2->forward(x)));
    capture(trace, prefix + ".ffn_hidden", hidden);
    x = x + down->forward(hidden);
    capture(trace, prefix + ".residual", x);
    return x;
  }
};

struct ProbeModel : torch::nn::Module {
  torch::nn::Embedding token{nullptr}, position{nullptr};
  torch::nn::LayerNorm final_norm{nullptr};
  torch::nn::Linear head{nullptr};
  std::vector<std::shared_ptr<Block>> blocks;
  explicit ProbeModel(const Options& o) {
    token = register_module("token", torch::nn::Embedding(o.vocab, o.width));
    position = register_module("position", torch::nn::Embedding(o.seq, o.width));
    for (int64_t i = 0; i < o.layers; ++i)
      blocks.push_back(register_module("block" + std::to_string(i), std::make_shared<Block>(o.width, o.heads)));
    final_norm = register_module("final_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({o.width})));
    head = register_module("head", torch::nn::Linear(o.width, o.vocab));
  }
  Tensor forward(const Tensor& ids, Trace* trace = nullptr) {
    auto positions = torch::arange(ids.size(1), ids.options());
    auto x = token->forward(ids) + position->forward(positions).unsqueeze(0);
    capture(trace, "embedding", x);
    for (size_t i = 0; i < blocks.size(); ++i) x = blocks[i]->forward(x, trace, "block" + std::to_string(i));
    auto logits = head->forward(final_norm->forward(x));
    capture(trace, "logits", logits);
    return logits;
  }
};

void copy_model(ProbeModel& destination, const ProbeModel& source) {
  at::NoGradGuard no_grad;
  auto src = source.named_parameters();
  for (auto& p : destination.named_parameters()) p.value().copy_(src[p.key()]);
}

Tensor objective(const Tensor& logits, const Tensor& targets) {
  // Cross entropy explicitly expressed to exercise log_softmax and gather backward.
  return -torch::log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).mean();
}

struct Batch { Tensor input, target; };

Batch make_batch(const Options& o, const torch::Device& device, bool repeated) {
  auto ids = torch::empty({o.batch, o.seq + 1}, torch::TensorOptions().dtype(torch::kLong));
  auto a = ids.accessor<int64_t, 2>();
  for (int64_t b = 0; b < o.batch; ++b)
    for (int64_t t = 0; t <= o.seq; ++t)
      a[b][t] = (b * 7 + (repeated ? 0 : t * 5 + (t * t) / 3)) % o.vocab;
  return {ids.slice(1, 0, o.seq).contiguous().to(device),
          ids.slice(1, 1, o.seq + 1).contiguous().to(device)};
}

torch::optim::AdamWOptions optimizer_options() {
  // The larger-than-default epsilon avoids magnifying near-zero roundoff in the
  // cross-device first-update comparison. Record this choice in every run.
  return torch::optim::AdamWOptions(0.003).eps(1e-5).weight_decay(0.01);
}

Tensor update(ProbeModel& model, torch::optim::AdamW& optimizer, const Batch& batch) {
  optimizer.zero_grad();
  auto loss = objective(model.forward(batch.input), batch.target);
  loss.backward();
  optimizer.step();
  return loss.detach();
}

void finite(const Tensor& value, const std::string& name) {
  require(value.defined(), name + " is undefined");
  require(torch::isfinite(value).all().item<bool>(), name + " contains non-finite values");
}

void compare(const std::string& name, const Tensor& actual, const Tensor& reference,
             double atol = 3e-5, double rtol = 3e-4) {
  finite(actual, name + " actual");
  finite(reference, name + " reference");
  auto a = actual.detach().to(torch::kCPU), r = reference.detach().to(torch::kCPU);
  require(a.sizes() == r.sizes(), name + " shape mismatch");
  auto diff = (a - r).abs();
  double error = diff.max().item<double>();
  double ratio = (diff / (atol + r.abs() * rtol)).max().item<double>();
  std::cout << "check=" << name << " max_abs=" << error << " tolerance_ratio=" << ratio
            << " atol=" << atol << " rtol=" << rtol << '\n';
  require(ratio <= 1.0, name + " exceeds numerical tolerance");
}

double loss_value(ProbeModel& model, const Batch& batch) {
  at::NoGradGuard no_grad;
  auto loss = objective(model.forward(batch.input), batch.target);
  finite(loss, "loss");
  return loss.item<double>();
}

void hip_info(const std::string& phase) {
  // Optional diagnostics from the runtime already loaded by LibTorch. Avoid a
  // compile-time HIP SDK dependency or a second separately selected runtime.
  using MemInfo = int (*)(size_t*, size_t*);
  using DeviceName = int (*)(char*, int, int);
  using Version = int (*)(int*);
  auto mem_info = reinterpret_cast<MemInfo>(dlsym(RTLD_DEFAULT, "hipMemGetInfo"));
  auto device_name = reinterpret_cast<DeviceName>(dlsym(RTLD_DEFAULT, "hipDeviceGetName"));
  auto version = reinterpret_cast<Version>(dlsym(RTLD_DEFAULT, "hipRuntimeGetVersion"));
  if (phase == "initial") {
    char name[256]{};
    int runtime_version = 0;
    if (device_name && device_name(name, sizeof(name), 0) == 0)
      std::cout << "hip_device_name=\"" << name << "\"\n";
    if (version && version(&runtime_version) == 0)
      std::cout << "hip_runtime_version=" << runtime_version << '\n';
  }
  if (mem_info) {
    size_t available = 0, total = 0;
    const int status = mem_info(&available, &total);
    std::cout << "hip_memory_phase=" << phase << " status=" << status;
    if (status == 0) std::cout << " free_bytes=" << available << " total_bytes=" << total;
    std::cout << '\n';
  } else {
    std::cout << "hip_memory_phase=" << phase << " status=unavailable\n";
  }
}

void memory(const torch::Device& device) {
  struct rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  std::cout << "process_peak_rss_kib=" << usage.ru_maxrss << '\n';
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line))
    if (line.rfind("VmRSS:", 0) == 0 || line.rfind("VmSwap:", 0) == 0)
      std::cout << "process_" << line << '\n';
  if (device.is_cuda()) hip_info("final");
}

void save_checkpoint(const std::string& file, ProbeModel& model, torch::optim::AdamW& optimizer) {
  torch::serialize::OutputArchive archive, model_archive, optimizer_archive;
  model.save(model_archive);
  optimizer.save(optimizer_archive);
  archive.write("model", model_archive);
  archive.write("optimizer", optimizer_archive);
  archive.save_to(file);
}

void load_checkpoint(const std::string& file, const torch::Device& device,
                     ProbeModel& model, torch::optim::AdamW& optimizer) {
  torch::serialize::InputArchive archive, model_archive, optimizer_archive;
  archive.load_from(file, device);
  archive.read("model", model_archive);
  archive.read("optimizer", optimizer_archive);
  model.load(model_archive);
  optimizer.load(optimizer_archive);
}

void verify(const Options& o, const torch::Device& device) {
  ProbeModel cpu(o), model(o);
  model.to(device);
  copy_model(model, cpu);
  const auto cpu_batch = make_batch(o, torch::kCPU, false);
  const auto batch = make_batch(o, device, false);
  torch::optim::AdamW cpu_optimizer(cpu.parameters(), optimizer_options());
  torch::optim::AdamW optimizer(model.parameters(), optimizer_options());
  Trace cpu_trace, trace;
  auto cpu_logits = cpu.forward(cpu_batch.input, &cpu_trace);
  auto logits = model.forward(batch.input, &trace);
  require(trace.size() == cpu_trace.size(), "Trace size mismatch");
  for (size_t i = 0; i < trace.size(); ++i) {
    require(trace[i].first == cpu_trace[i].first, "Trace name mismatch");
    compare("forward." + trace[i].first, trace[i].second, cpu_trace[i].second);
    auto value = trace[i].second.detach().to(torch::kCPU);
    std::cout << "inspection=" << trace[i].first << " elements=" << value.numel()
              << " mean=" << value.mean().item<double>()
              << " rms=" << value.square().mean().sqrt().item<double>()
              << " min=" << value.min().item<double>() << " max=" << value.max().item<double>() << '\n';
  }
  auto cpu_loss = objective(cpu_logits, cpu_batch.target), loss = objective(logits, batch.target);
  compare("loss", loss, cpu_loss);
  cpu_loss.backward();
  loss.backward();
  auto cpu_parameters = cpu.named_parameters();
  double gradient_norm_squared = 0;
  for (auto& p : model.named_parameters()) {
    compare("gradient." + p.key(), p.value().grad(), cpu_parameters[p.key()].grad());
    gradient_norm_squared += p.value().grad().square().sum().item<double>();
  }
  require(gradient_norm_squared > 1e-12, "All gradients are zero");
  auto before = model.head->weight.detach().clone();
  cpu_optimizer.step();
  optimizer.step();
  for (auto& p : model.named_parameters())
    compare("adamw." + p.key(), p.value(), cpu_parameters[p.key()]);
  const double change = (model.head->weight - before).abs().max().item<double>();
  require(change > 0, "Optimizer did not change output weights");
  std::cout << "optimizer_max_head_change=" << change
            << " gradient_l2=" << std::sqrt(gradient_norm_squared) << '\n';
  cpu_trace.clear(); trace.clear();
  cpu_logits = Tensor(); logits = Tensor(); cpu_loss = Tensor(); loss = Tensor();

  {
    at::NoGradGuard no_grad;
    auto baseline = model.forward(batch.input);
    auto changed_input = batch.input.clone();
    changed_input.select(1, o.seq - 1).add_(1).remainder_(o.vocab);
    auto changed_output = model.forward(changed_input);
    compare("causal_prefix", changed_output.slice(1, 0, o.seq - 1), baseline.slice(1, 0, o.seq - 1), 1e-6, 1e-5);
    auto original_bias = model.head->bias.detach().clone();
    model.head->bias.select(0, 0).add_(0.75);
    auto expected = baseline.clone();
    expected.select(-1, 0).add_(0.75);
    compare("direct_edit", model.forward(batch.input), expected, 1e-6, 1e-5);
    model.head->bias.copy_(original_bias);
    compare("edit_restored", model.forward(batch.input), baseline, 1e-7, 1e-6);
  }

  const auto practice = make_batch(o, device, true);
  const auto initial = loss_value(model, practice);
  for (int64_t i = 0; i < o.steps; ++i) {
    auto step_loss = update(model, optimizer, practice);
    finite(step_loss, "training loss");
  }
  const auto final = loss_value(model, practice);
  std::cout << "learning_task=repeat_symbol initial_loss=" << initial << " final_loss=" << final
            << " steps=" << o.steps << '\n';
  require(final < initial * 0.8, "Training did not reduce repeat-symbol loss by at least 20%");
  for (auto& p : model.named_parameters()) finite(p.value(), "trained." + p.key());

  const auto file = (std::filesystem::path(o.output) / "verified_checkpoint.pt").string();
  save_checkpoint(file, model, optimizer);
  ProbeModel restored(o);
  restored.to(device);
  torch::optim::AdamW restored_optimizer(restored.parameters(), optimizer_options());
  load_checkpoint(file, device, restored, restored_optimizer);
  {
    at::NoGradGuard no_grad;
    compare("checkpoint_output", restored.forward(practice.input), model.forward(practice.input), 1e-7, 1e-6);
  }
  auto continued_loss = update(model, optimizer, practice);
  auto restored_loss = update(restored, restored_optimizer, practice);
  compare("checkpoint_next_loss", restored_loss, continued_loss, 1e-7, 1e-6);
  auto parameters = model.named_parameters();
  for (auto& p : restored.named_parameters())
    compare("checkpoint_next_update." + p.key(), p.value(), parameters[p.key()], 1e-7, 1e-6);
  sync(device);
  std::cout << "checkpoint=" << file << '\n';
  memory(device);
  std::cout << "result=PASS scope=" << (device.is_cuda() ? "cpu_gpu_training_parity" : "cpu_only") << std::endl;
}

void benchmark(const Options& o, const torch::Device& device) {
  ProbeModel model(o);
  model.to(device);
  torch::optim::AdamW optimizer(model.parameters(), optimizer_options());
  const auto batch = make_batch(o, device, true);
  const auto initial = loss_value(model, batch);
  sync(device);
  auto warmup_start = Clock::now();
  for (int64_t i = 0; i < o.warmup; ++i) finite(update(model, optimizer, batch), "warmup loss");
  sync(device);
  double warmup_seconds = std::chrono::duration<double>(Clock::now() - warmup_start).count();
  std::cout << "warmup_steps=" << o.warmup << " warmup_seconds=" << warmup_seconds << std::endl;
  std::vector<double> times;
  const auto start = Clock::now();
  Tensor last_loss;
  for (int64_t step = 0; step < o.steps; ++step) {
    const auto step_start = Clock::now();
    last_loss = update(model, optimizer, batch);
    sync(device);
    times.push_back(std::chrono::duration<double>(Clock::now() - step_start).count());
    if ((step + 1) % 10 == 0) {
      finite(last_loss, "benchmark loss");
      std::cout << "progress_steps=" << step + 1 << " loss=" << last_loss.item<double>() << std::endl;
    }
    if (o.seconds > 0 && std::chrono::duration<double>(Clock::now() - start).count() >= o.seconds) break;
  }
  sync(device);
  const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  finite(last_loss, "benchmark final loss");
  auto sorted = times;
  std::sort(sorted.begin(), sorted.end());
  const auto percentile = [&](double p) { return sorted[std::min(sorted.size() - 1, size_t(std::ceil(p * sorted.size()) - 1))]; };
  const auto final = loss_value(model, batch);
  for (auto& p : model.named_parameters()) {
    finite(p.value(), "benchmark parameter " + p.key());
    finite(p.value().grad(), "benchmark gradient " + p.key());
  }
  std::cout << "measured_steps=" << times.size() << " measured_seconds=" << elapsed
            << " tokens_per_second=" << (double(times.size()) * o.batch * o.seq / elapsed)
            << " step_seconds_mean=" << std::accumulate(times.begin(), times.end(), 0.0) / times.size()
            << " step_seconds_p50=" << percentile(0.5) << " step_seconds_p95=" << percentile(0.95)
            << " initial_loss=" << initial << " final_loss=" << final << '\n';
  // Benchmarks leave a compact timing artifact; the verified model+optimizer
  // checkpoint is produced by verify, avoiding large incidental capacity files.
  std::ofstream samples(std::filesystem::path(o.output) / "step_times.tsv");
  require(bool(samples), "Cannot write timing artifact");
  samples << "step\tseconds\n" << std::setprecision(10);
  for (size_t i = 0; i < times.size(); ++i) samples << i + 1 << '\t' << times[i] << '\n';
  memory(device);
  std::cout << "result=PASS scope=bounded_training_benchmark" << std::endl;
}
} // namespace

int main(int argc, char** argv) {
  try {
    std::cout << std::setprecision(10) << std::unitbuf;
    const auto o = parse(argc, argv);
    at::set_num_threads(o.threads);
    at::set_num_interop_threads(1);
    at::manual_seed(o.seed);
    const torch::Device device(o.device == "cuda" ? torch::kCUDA : torch::kCPU, o.device == "cuda" ? 0 : -1);
    if (device.is_cuda()) require(torch::cuda::is_available(), "CUDA/HIP runtime unavailable; refusing CPU fallback");
    if (device.is_cuda()) {
      auto initialization = torch::zeros({1}, torch::TensorOptions().device(device));
      sync(device);
      hip_info("initial");
    }
    std::filesystem::create_directories(o.output);
    int64_t parameters = 0;
    {
      // Count using the known shapes instead of constructing another large model.
      const int64_t d = o.width;
      parameters = o.vocab * d + o.seq * d + o.layers * (12 * d * d + 10 * d) + 2 * d + d * o.vocab + o.vocab;
    }
    std::ofstream config(std::filesystem::path(o.output) / "config.txt");
    require(bool(config), "Cannot write configuration artifact");
    const auto write_config = [&](std::ostream& out) {
      out << "torch_version=" << TORCH_VERSION << " mode=" << o.mode << " device=" << device
          << " dtype=float32 width=" << o.width << " layers=" << o.layers << " heads=" << o.heads
          << " seq=" << o.seq << " batch=" << o.batch << " vocab=" << o.vocab
          << " steps=" << o.steps << " warmup=" << o.warmup << " seconds=" << o.seconds
          << " seed=" << o.seed << " threads=" << o.threads << " parameters=" << parameters
          << " optimizer=AdamW lr=0.003 eps=1e-5 weight_decay=0.01\n";
      out << "parameter_bytes=" << parameters * 4 << " parameter_gradient_adam_state_bytes=" << parameters * 16
          << " memory_note=excludes_activations_workspace_runtime_and_allocator\n";
    };
    write_config(std::cout); write_config(config); config.close();
    if (o.mode == "verify") verify(o, device);
    else benchmark(o, device);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "result=FAIL error=" << e.what() << std::endl;
    return 1;
  }
}
