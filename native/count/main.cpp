// Copyright (C) 2020-2026, François-Guillaume Fernandez.
// Licensed under the Apache License 2.0. See LICENSE for details.
#include <filesystem>
#include <fstream>
#include <iostream>

#include "count.hpp"

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  using namespace torchscan;
  const auto usage = "Usage: torchscan-count capture.json [-o report.json]";
  try {
    auto path = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path();
    if (argc == 2 && (path == "--help" || path == "-h")) {
      std::cout << usage
                << "\nCounts saved operator metadata; never runs a model.\nExit codes: 0 complete, 2 partial or "
                   "invalid input.\n";
      return 0;
    }
    if (argc != 2 && !(argc == 4 && std::filesystem::path(argv[2]) == "-o")) throw std::invalid_argument(usage);
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open capture file.");
    std::string contents((std::istreambuf_iterator<char>(file)), {});
    if (file.bad()) throw std::runtime_error("Cannot read capture file.");
    auto report = analyze(parse_json(contents));
    std::ofstream output;
    if (argc == 4) {
      output.open(std::filesystem::path(argv[3]), std::ios::binary);
      if (!output) throw std::runtime_error("Cannot open report output.");
    }
    auto& stream = argc == 4 ? output : std::cout;
    write_json(stream, report);
    stream << '\n';
    stream.flush();
    if (!stream) throw std::runtime_error("Cannot write report.");
    if (report["total"]["status"] == "complete") return 0;
    std::cerr << "Incomplete count; see report diagnostics.\n";
  } catch (const std::exception& error) {
    std::cerr << "torchscan-count: " << error.what() << '\n';
  }
  return 2;
}
