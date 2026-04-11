// cdf_lookup.cpp
// Compile:
//   g++ -O3 -std=c++17 -march=native -DNDEBUG cdf_lookup.cpp -o cdf_lookup
//
// Run (example):
//   ./cdf_lookup out/cdf_grid.f32 out/cdf_grid_meta.json 120 50.0 1000000
// Arguments:
//   1) path to grid file (cdf_grid.f32)
//   2) path to meta json (cdf_grid_meta.json)
//   3) lag (int)
//   4) bp value (double, in bp units)
//   5) iterations (optional, default 1) — repeats lookup to measure time reliably
//
// Output:
//   cdf=<value>
//   avg_ns=<nanoseconds per lookup over iterations>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// --- Minimal JSON "parser" for our meta file: extracts integers/doubles by key.
// This avoids external deps. It assumes the meta JSON is simple and contains keys like:
// "shape": [901, 90001], "bp10_min": -40000, "bp10_max": 50000
static bool extract_int64(const std::string& s, const char* key, int64_t& out) {
  std::string k = std::string("\"") + key + "\"";
  size_t pos = s.find(k);
  if (pos == std::string::npos) return false;
  pos = s.find(':', pos);
  if (pos == std::string::npos) return false;
  pos++;
  while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) pos++;

  // parse integer (could be negative)
  char* endptr = nullptr;
  errno = 0;
  long long val = std::strtoll(s.c_str() + pos, &endptr, 10);
  if (errno != 0 || endptr == (s.c_str() + pos)) return false;
  out = (int64_t)val;
  return true;
}

static bool extract_shape(const std::string& s, int64_t& rows, int64_t& cols) {
  std::string k = "\"shape\"";
  size_t pos = s.find(k);
  if (pos == std::string::npos) return false;
  pos = s.find('[', pos);
  if (pos == std::string::npos) return false;
  pos++;
  while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) pos++;

  char* endptr = nullptr;
  errno = 0;
  long long r = std::strtoll(s.c_str() + pos, &endptr, 10);
  if (errno != 0 || endptr == (s.c_str() + pos)) return false;

  pos = s.find(',', (size_t)(endptr - s.c_str()));
  if (pos == std::string::npos) return false;
  pos++;
  while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) pos++;

  errno = 0;
  long long c = std::strtoll(s.c_str() + pos, &endptr, 10);
  if (errno != 0 || endptr == (s.c_str() + pos)) return false;

  rows = (int64_t)r;
  cols = (int64_t)c;
  return true;
}

static std::string read_file_to_string(const char* path) {
  FILE* fp = std::fopen(path, "rb");
  if (!fp) {
    std::perror("fopen");
    std::exit(1);
  }
  std::fseek(fp, 0, SEEK_END);
  long n = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  std::string s;
  s.resize((size_t)n);
  if (n > 0) {
    size_t got = std::fread(&s[0], 1, (size_t)n, fp);
    if (got != (size_t)n) {
      std::perror("fread");
      std::exit(1);
    }
  }
  std::fclose(fp);
  return s;
}

static inline int64_t bp_to_bp10(double bp) {
  // round(bp * 10) as int64
  // using nearbyint for ties-to-even; matches Python's round-ish behavior well enough.
  // If you need strict "round half away from zero", we can change this.
  double x = bp * 10.0;
  return (int64_t)llround(x);
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage:\n  " << argv[0]
              << " <cdf_grid.f32> <cdf_grid_meta.json> <lag> <bp> [iterations]\n";
    return 1;
  }

  const char* grid_path = argv[1];
  const char* meta_path = argv[2];
  int lag = std::atoi(argv[3]);
  double bp = std::atof(argv[4]);
  long long iters = (argc >= 6) ? std::atoll(argv[5]) : 1;
  if (iters <= 0) iters = 1;

  // Load meta
  std::string meta = read_file_to_string(meta_path);

  int64_t bp10_min = 0, bp10_max = 0, rows = 0, cols = 0;
  if (!extract_int64(meta, "bp10_min", bp10_min)) {
    std::cerr << "Failed to parse bp10_min from meta\n";
    return 1;
  }
  if (!extract_int64(meta, "bp10_max", bp10_max)) {
    std::cerr << "Failed to parse bp10_max from meta\n";
    return 1;
  }
  if (!extract_shape(meta, rows, cols)) {
    std::cerr << "Failed to parse shape from meta\n";
    return 1;
  }

  if (lag < 1 || lag >= rows) {
    std::cerr << "Lag out of range. lag=" << lag << " rows=" << rows << "\n";
    return 1;
  }

  // mmap grid
  int fd = ::open(grid_path, O_RDONLY);
  if (fd < 0) {
    std::perror("open");
    return 1;
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    std::perror("fstat");
    return 1;
  }

  size_t expected_bytes = (size_t)rows * (size_t)cols * sizeof(float);
  if ((size_t)st.st_size < expected_bytes) {
    std::cerr << "Grid file too small. size=" << (size_t)st.st_size
              << " expected>=" << expected_bytes << "\n";
    return 1;
  }

  void* addr = mmap(nullptr, expected_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
    std::perror("mmap");
    return 1;
  }
  ::close(fd);

  const float* grid = static_cast<const float*>(addr);

  // Compute index
  int64_t bp10 = bp_to_bp10(bp);
  float result = 0.0f;

  auto lookup_once = [&]() -> float {
    if (bp10 < bp10_min) return 0.0f;
    if (bp10 > bp10_max) return 1.0f;
    int64_t col = bp10 - bp10_min; // 0..cols-1
    // row-major: offset = lag * cols + col
    return grid[(int64_t)lag * cols + col];
  };

  // Timing (repeat to get stable measurement)
  // Use volatile to prevent the compiler from optimizing away repeated lookups.
  volatile float sink = 0.0f;

  auto t0 = std::chrono::steady_clock::now();
  for (long long i = 0; i < iters; i++) {
    sink = lookup_once();
  }
  auto t1 = std::chrono::steady_clock::now();

  // Ensure we print a real result (from last lookup)
  result = (float)sink;

  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  double avg_ns = (double)ns / (double)iters;

  std::cout.setf(std::ios::fixed);
  std::cout.precision(8);
  std::cout << "cdf=" << result << "\n";
  std::cout.precision(2);
  std::cout << "avg_ns=" << avg_ns << "\n";

  // Cleanup
  munmap(addr, expected_bytes);
  return 0;
}
