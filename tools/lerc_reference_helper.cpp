#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern "C" {
using lerc_status = unsigned int;

lerc_status lerc_getBlobInfo(
    const unsigned char* pLercBlob,
    unsigned int blobSize,
    unsigned int* infoArray,
    double* dataRangeArray,
    int infoArraySize,
    int dataRangeArraySize);

lerc_status lerc_decode_4D(
    const unsigned char* pLercBlob,
    unsigned int blobSize,
    int nMasks,
    unsigned char* pValidBytes,
    int nDepth,
    int nCols,
    int nRows,
    int nBands,
    unsigned int dataType,
    void* pData,
    unsigned char* pUsesNoData,
    double* noDataValues);
}

namespace {

constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;

struct BlobInfo {
  unsigned int version = 0;
  unsigned int data_type = 0;
  int depth = 0;
  int width = 0;
  int height = 0;
  int bands = 0;
  unsigned int valid_pixel_count = 0;
  unsigned int blob_size = 0;
  int mask_count = 0;
  bool uses_no_data_value = false;
  double z_min = 0;
  double z_max = 0;
  double max_z_error = 0;
};

struct DecodeResult {
  std::vector<std::uint8_t> pixel_bytes;
  std::vector<int> pixel_shape;
  std::optional<std::vector<std::uint8_t>> mask_bytes;
  std::optional<std::vector<int>> mask_shape;
  double checksum = 0;
  std::uint64_t valid_sum = 0;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

std::vector<std::uint8_t> read_blob(const std::string& path) {
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".csv") {
    std::ifstream in(path);
    if (!in) {
      fail("failed to open blob file: " + path);
    }

    std::vector<std::uint8_t> bytes;
    std::string token;
    while (std::getline(in, token, ',')) {
      std::size_t start = token.find_first_not_of(" \t\r\n");
      if (start == std::string::npos) {
        continue;
      }
      std::size_t end = token.find_last_not_of(" \t\r\n");
      auto trimmed = token.substr(start, end - start + 1);
      unsigned long value = std::stoul(trimmed);
      if (value > 255) {
        fail("csv byte value out of range: " + trimmed);
      }
      bytes.push_back(static_cast<std::uint8_t>(value));
    }
    return bytes;
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    fail("failed to open blob file: " + path);
  }
  in.seekg(0, std::ios::end);
  auto size = in.tellg();
  if (size < 0) {
    fail("failed to determine blob size: " + path);
  }
  in.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (!bytes.empty()) {
    in.read(reinterpret_cast<char*>(bytes.data()), size);
  }
  if (!in) {
    fail("failed to read blob file: " + path);
  }
  return bytes;
}

std::size_t data_type_size(unsigned int data_type) {
  switch (data_type) {
    case 0:
    case 1:
      return 1;
    case 2:
    case 3:
      return 2;
    case 4:
    case 5:
    case 6:
      return 4;
    case 7:
      return 8;
    default:
      fail("unsupported LERC data type code: " + std::to_string(data_type));
  }
}

BlobInfo get_blob_info(const std::vector<std::uint8_t>& blob) {
  unsigned int info[11] = {};
  double ranges[3] = {};
  auto status = lerc_getBlobInfo(
      blob.data(),
      static_cast<unsigned int>(blob.size()),
      info,
      ranges,
      11,
      3);
  if (status != 0) {
    fail("lerc_getBlobInfo failed with status " + std::to_string(status));
  }

  BlobInfo out;
  out.version = info[0];
  out.data_type = info[1];
  out.depth = static_cast<int>(info[2]);
  out.width = static_cast<int>(info[3]);
  out.height = static_cast<int>(info[4]);
  out.bands = static_cast<int>(info[5]);
  out.valid_pixel_count = info[6];
  out.blob_size = info[7];
  out.mask_count = static_cast<int>(info[8]);
  out.uses_no_data_value = info[10] != 0;
  out.z_min = ranges[0];
  out.z_max = ranges[1];
  out.max_z_error = ranges[2];
  return out;
}

std::size_t raw_sample_index(
    int band,
    int row,
    int col,
    int depth,
    int rows,
    int cols,
    int n_depth) {
  return (((static_cast<std::size_t>(band) * static_cast<std::size_t>(rows) +
            static_cast<std::size_t>(row)) *
               static_cast<std::size_t>(cols) +
           static_cast<std::size_t>(col)) *
              static_cast<std::size_t>(n_depth)) +
         static_cast<std::size_t>(depth);
}

std::vector<int> pixel_shape_for(const BlobInfo& info) {
  if (info.bands <= 1 && info.depth <= 1) {
    return {info.height, info.width};
  }
  if (info.bands <= 1) {
    return {info.height, info.width, info.depth};
  }
  if (info.depth <= 1) {
    return {info.height, info.width, info.bands};
  }
  return {info.height, info.width, info.bands, info.depth};
}

std::vector<std::uint8_t> normalize_pixels(
    const std::vector<std::uint8_t>& raw,
    const std::vector<std::uint8_t>& raw_masks,
    const BlobInfo& info) {
  const auto element_size = data_type_size(info.data_type);
  const std::size_t pixel_count =
      static_cast<std::size_t>(info.height) * static_cast<std::size_t>(info.width);
  std::vector<std::uint8_t> normalized;
  normalized.reserve(raw.size());

  for (int row = 0; row < info.height; ++row) {
    for (int col = 0; col < info.width; ++col) {
      const std::size_t pixel =
          static_cast<std::size_t>(row) * static_cast<std::size_t>(info.width) +
          static_cast<std::size_t>(col);
      if (info.bands <= 1) {
        const bool valid =
            info.mask_count == 0 || raw_masks[pixel] != 0;
        for (int depth = 0; depth < std::max(info.depth, 1); ++depth) {
          auto raw_index = raw_sample_index(0, row, col, depth, info.height, info.width, std::max(info.depth, 1));
          auto offset = raw_index * element_size;
          if (valid) {
            normalized.insert(
                normalized.end(),
                raw.begin() + static_cast<std::ptrdiff_t>(offset),
                raw.begin() + static_cast<std::ptrdiff_t>(offset + element_size));
          } else {
            normalized.insert(normalized.end(), element_size, 0);
          }
        }
        continue;
      }

      for (int band = 0; band < info.bands; ++band) {
        const bool valid =
            info.mask_count == 0 ||
            (info.mask_count == 1
                 ? raw_masks[pixel] != 0
                 : raw_masks[static_cast<std::size_t>(band) * pixel_count + pixel] != 0);
        for (int depth = 0; depth < std::max(info.depth, 1); ++depth) {
          auto raw_index = raw_sample_index(
              band,
              row,
              col,
              depth,
              info.height,
              info.width,
              std::max(info.depth, 1));
          auto offset = raw_index * element_size;
          if (valid) {
            normalized.insert(
                normalized.end(),
                raw.begin() + static_cast<std::ptrdiff_t>(offset),
                raw.begin() + static_cast<std::ptrdiff_t>(offset + element_size));
          } else {
            normalized.insert(normalized.end(), element_size, 0);
          }
        }
      }
    }
  }

  return normalized;
}

std::optional<std::vector<int>> mask_shape_for(const BlobInfo& info) {
  if (info.mask_count == 0) {
    return std::nullopt;
  }
  if (info.bands <= 1) {
    return std::vector<int>{info.height, info.width};
  }
  return std::vector<int>{info.height, info.width, info.bands};
}

std::optional<std::vector<std::uint8_t>> normalize_masks(
    const std::vector<std::uint8_t>& raw_masks,
    const BlobInfo& info) {
  if (info.mask_count == 0) {
    return std::nullopt;
  }

  const std::size_t pixel_count =
      static_cast<std::size_t>(info.height) * static_cast<std::size_t>(info.width);

  if (info.bands <= 1) {
    return std::vector<std::uint8_t>(raw_masks.begin(), raw_masks.begin() + static_cast<std::ptrdiff_t>(pixel_count));
  }

  std::vector<std::uint8_t> normalized;
  normalized.reserve(pixel_count * static_cast<std::size_t>(info.bands));

  for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
    for (int band = 0; band < info.bands; ++band) {
      if (info.mask_count == 1) {
        normalized.push_back(raw_masks[pixel]);
      } else {
        normalized.push_back(
            raw_masks[static_cast<std::size_t>(band) * pixel_count + pixel]);
      }
    }
  }

  return normalized;
}

template <typename T>
double checksum_values(const std::vector<std::uint8_t>& raw) {
  const std::size_t count = raw.size() / sizeof(T);
  double sum = 0;
  for (std::size_t i = 0; i < count; ++i) {
    T value{};
    std::memcpy(
        &value,
        raw.data() + static_cast<std::ptrdiff_t>(i * sizeof(T)),
        sizeof(T));
    sum += static_cast<double>(value);
  }
  return sum;
}

double checksum_for(const std::vector<std::uint8_t>& raw, unsigned int data_type) {
  switch (data_type) {
    case 0:
      return checksum_values<std::int8_t>(raw);
    case 1:
      return checksum_values<std::uint8_t>(raw);
    case 2:
      return checksum_values<std::int16_t>(raw);
    case 3:
      return checksum_values<std::uint16_t>(raw);
    case 4:
      return checksum_values<std::int32_t>(raw);
    case 5:
      return checksum_values<std::uint32_t>(raw);
    case 6:
      return checksum_values<float>(raw);
    case 7:
      return checksum_values<double>(raw);
    default:
      fail("unsupported LERC data type code: " + std::to_string(data_type));
  }
}

std::string fnv1a64(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t hash = kFnvOffset;
  for (auto byte : bytes) {
    hash ^= static_cast<std::uint64_t>(byte);
    hash *= kFnvPrime;
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << hash;
  return out.str();
}

DecodeResult decode_blob(const std::vector<std::uint8_t>& blob) {
  const auto info = get_blob_info(blob);
  const auto element_size = data_type_size(info.data_type);
  const int n_depth = std::max(info.depth, 1);
  const std::size_t sample_count =
      static_cast<std::size_t>(info.width) * static_cast<std::size_t>(info.height) *
      static_cast<std::size_t>(n_depth) * static_cast<std::size_t>(std::max(info.bands, 1));
  const std::size_t raw_size = sample_count * element_size;
  const std::size_t aligned_len =
      (raw_size + sizeof(double) - 1) / sizeof(double);

  std::vector<double> aligned_raw(aligned_len);
  std::vector<std::uint8_t> raw_masks;
  if (info.mask_count > 0) {
    raw_masks.resize(
        static_cast<std::size_t>(info.width) * static_cast<std::size_t>(info.height) *
        static_cast<std::size_t>(info.mask_count));
  }
  std::vector<std::uint8_t> uses_no_data;
  std::vector<double> no_data_values;
  if (info.uses_no_data_value) {
    uses_no_data.resize(static_cast<std::size_t>(std::max(info.bands, 1)));
    no_data_values.resize(static_cast<std::size_t>(std::max(info.bands, 1)));
  }

  auto status = lerc_decode_4D(
      blob.data(),
      static_cast<unsigned int>(blob.size()),
      info.mask_count,
      raw_masks.empty() ? nullptr : raw_masks.data(),
      n_depth,
      info.width,
      info.height,
      std::max(info.bands, 1),
      info.data_type,
      aligned_raw.data(),
      uses_no_data.empty() ? nullptr : uses_no_data.data(),
      no_data_values.empty() ? nullptr : no_data_values.data());
  if (status != 0) {
    fail("lerc_decode_4D failed with status " + std::to_string(status));
  }

  const auto* raw_ptr =
      reinterpret_cast<const std::uint8_t*>(aligned_raw.data());
  std::vector<std::uint8_t> raw(raw_ptr, raw_ptr + static_cast<std::ptrdiff_t>(raw_size));

  DecodeResult result;
  result.pixel_shape = pixel_shape_for(info);
  result.pixel_bytes = normalize_pixels(raw, raw_masks, info);
  result.mask_shape = mask_shape_for(info);
  result.mask_bytes = normalize_masks(raw_masks, info);
  result.checksum = checksum_for(raw, info.data_type);
  result.valid_sum = 0;
  if (result.mask_bytes.has_value()) {
    for (auto value : *result.mask_bytes) {
      result.valid_sum += static_cast<std::uint64_t>(value);
    }
  }
  return result;
}

void print_json_array(const std::vector<int>& values) {
  std::cout << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ',';
    }
    std::cout << values[i];
  }
  std::cout << ']';
}

void print_metadata_json(const BlobInfo& info) {
  std::cout << '{'
            << "\"version\":" << info.version << ','
            << "\"data_type\":" << info.data_type << ','
            << "\"width\":" << info.width << ','
            << "\"height\":" << info.height << ','
            << "\"depth\":" << info.depth << ','
            << "\"band_count\":" << info.bands << ','
            << "\"valid_pixel_count\":" << info.valid_pixel_count << ','
            << "\"blob_size\":" << info.blob_size << ','
            << "\"mask_count\":" << info.mask_count << ','
            << "\"uses_no_data_value\":" << (info.uses_no_data_value ? "true" : "false") << ','
            << "\"z_min\":" << std::setprecision(17) << info.z_min << ','
            << "\"z_max\":" << std::setprecision(17) << info.z_max << ','
            << "\"max_z_error\":" << std::setprecision(17) << info.max_z_error
            << "}\n";
}

void print_hash_json(const DecodeResult& result) {
  std::cout << '{'
            << "\"pixel_shape\":";
  print_json_array(result.pixel_shape);
  std::cout << ','
            << "\"pixel_byte_len\":" << result.pixel_bytes.size() << ','
            << "\"pixel_hash\":\"" << fnv1a64(result.pixel_bytes) << "\","
            << "\"mask_shape\":";
  if (result.mask_shape.has_value()) {
    print_json_array(*result.mask_shape);
  } else {
    std::cout << "null";
  }
  std::cout << ','
            << "\"mask_byte_len\":";
  if (result.mask_bytes.has_value()) {
    std::cout << result.mask_bytes->size();
  } else {
    std::cout << "null";
  }
  std::cout << ','
            << "\"mask_hash\":";
  if (result.mask_bytes.has_value()) {
    std::cout << '"' << fnv1a64(*result.mask_bytes) << '"';
  } else {
    std::cout << "null";
  }
  std::cout << "}\n";
}

void print_benchmark_json(
    const DecodeResult& result,
    std::size_t iterations,
    double total_seconds) {
  std::cout << '{'
            << "\"iterations\":" << iterations << ','
            << "\"total_seconds\":" << std::setprecision(17) << total_seconds << ','
            << "\"checksum\":" << std::setprecision(17)
            << (result.checksum * static_cast<double>(iterations)) << ','
            << "\"valid_sum\":" << (result.valid_sum * iterations) << ','
            << "\"pixel_byte_len\":" << result.pixel_bytes.size() << ','
            << "\"pixel_hash\":\"" << fnv1a64(result.pixel_bytes) << "\""
            << "}\n";
}

std::size_t parse_iterations(const std::vector<std::string_view>& args) {
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == "--iterations") {
      return static_cast<std::size_t>(std::stoull(std::string(args[i + 1])));
    }
  }
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      fail("usage: lerc-reference <metadata|hash|benchmark> <blob-path> [--iterations N]");
    }

    const std::string command = argv[1];
    const std::string blob_path = argv[2];
    const auto blob = read_blob(blob_path);

    if (command == "metadata") {
      print_metadata_json(get_blob_info(blob));
      return 0;
    }

    if (command == "hash") {
      print_hash_json(decode_blob(blob));
      return 0;
    }

    if (command == "benchmark") {
      std::vector<std::string_view> args;
      for (int i = 3; i < argc; ++i) {
        args.emplace_back(argv[i]);
      }
      const auto iterations = parse_iterations(args);
      DecodeResult last_result;
      auto start = std::chrono::steady_clock::now();
      for (std::size_t i = 0; i < iterations; ++i) {
        last_result = decode_blob(blob);
      }
      auto elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start);
      print_benchmark_json(last_result, iterations, elapsed.count());
      return 0;
    }

    fail("unknown command: " + command);
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
  }
}
