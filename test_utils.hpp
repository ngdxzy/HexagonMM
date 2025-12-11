#pragma once

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "loadTensor.hpp"
#include "helperFunction.hpp"
#include "errorMetrics.hpp"
#include "ggml-impl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"

#define GGML_LOG_INFO(...) fprintf(stderr, __VA_ARGS__)
#define GGML_LOG_ERROR(...) fprintf(stderr, "ERROR: " __VA_ARGS__)


bool initialize_backend(ggml_backend_t& backend);
void print_tensor(const char* name, struct ggml_tensor* tensor, int max_elements = 10);


/// \brief print the matrix
/// \param matrix the matrix
/// \param n_cols the number of columns
/// \param n_printable_rows the number of printable rows
/// \param n_printable_cols the number of printable columns
/// \param ostream the output stream
/// \param col_sep the column separator
/// \param elide_sym the elide symbol
/// \param w the width
template <typename T>
inline void print_matrix(T* matrix, int n_rows, int n_cols,
                  int n_printable_rows = 10, int n_printable_cols = 10,
                  std::ostream &ostream = std::cout,
                  const char col_sep[] = "  ", const char elide_sym[] = " ... ",
                  int w = -1) {

  if (w == -1) {
    w = 6;
  }
  
  n_printable_rows = std::min(n_rows, n_printable_rows);
  n_printable_cols = std::min(n_cols, n_printable_cols);

  const bool elide_rows = n_printable_rows < n_rows;
  const bool elide_cols = n_printable_cols < n_cols;

  if (elide_rows || elide_cols) {
    w = std::max((int)w, (int)strlen(elide_sym));
  }

  w += 3; // for decimal point and two decimal digits
  ostream << std::fixed << std::setprecision(2);

#define print_row(what)                                                        \
  for (int col = 0; col < (n_printable_cols + 1) / 2; col++) {                 \
    ostream << std::right << std::setw(w) << std::scientific << std::setprecision(2) << (what);                           \
    ostream << std::setw(0) << col_sep;                                        \
  }                                                                            \
  if (elide_cols) {                                                            \
    ostream << std::setw(0) << elide_sym;                                      \
  }                                                                            \
  for (int i = 0; i < n_printable_cols / 2; i++) {                             \
    [[maybe_unused]]int col = n_cols - n_printable_cols / 2 + i;                               \
    ostream << std::right << std::setw(w) << std::scientific << std::setprecision(2) << (what);                           \
    ostream << std::setw(0) << col_sep;                                        \
  }

  for (int row = 0; row < (n_printable_rows + 1) / 2; row++) {
    print_row(matrix[row * n_cols + col]);
    ostream << std::endl;
  }
  if (elide_rows) {
    print_row(elide_sym);
    ostream << std::endl;
  }
  for (int i = 0; i < n_printable_rows / 2; i++) {
    int row = n_rows - n_printable_rows / 2 + i;
    print_row(matrix[row * n_cols + col]);
    ostream << std::endl;
  }

#undef print_row
}
