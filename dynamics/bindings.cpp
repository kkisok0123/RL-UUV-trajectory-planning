#define FISH_DYNAMICS_PYBIND 1
#include "dynamics_step.cpp"

#include <cstring>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

void require_vector(const py::array &array, py::ssize_t expected, const char *name) {
  if (array.ndim() != 1 || array.shape(0) != expected) {
    std::string message =
        std::string(name) + " must have shape (" + std::to_string(expected) + ",)";
    throw std::invalid_argument(message);
  }
}

} // namespace

static py::tuple py_step(py::array_t<double, py::array::c_style | py::array::forcecast> state,
                         py::array_t<double, py::array::c_style | py::array::forcecast> body_params,
                         py::array_t<double, py::array::c_style | py::array::forcecast> fin_params,
                         double dt, double t_k,
                         py::array_t<double, py::array::c_style | py::array::forcecast> action_ref,
                         py::array_t<double, py::array::c_style | py::array::forcecast> hist,
                         double c_A, double fin_f) {
  require_vector(state, NX, "state");
  require_vector(body_params, 36, "body_params");
  require_vector(fin_params, 38, "fin_params");
  require_vector(action_ref, 5, "action_ref");
  require_vector(hist, 5, "hist");

  if (dt < 0.0) {
    throw std::invalid_argument("dt must be non-negative");
  }
  if (fin_f <= 0.0) {
    throw std::invalid_argument("fin_f must be positive");
  }

  double state_buf[NX];
  double bp_buf[36];
  double fp_buf[38];
  double ref_buf[5];
  double hist_buf[5];
  double hist_out[5];

  std::memcpy(state_buf, state.data(), sizeof(state_buf));
  std::memcpy(bp_buf, body_params.data(), sizeof(bp_buf));
  std::memcpy(fp_buf, fin_params.data(), sizeof(fp_buf));
  std::memcpy(ref_buf, action_ref.data(), sizeof(ref_buf));
  std::memcpy(hist_buf, hist.data(), sizeof(hist_buf));

  simulate_step_arrays(state_buf, bp_buf, fp_buf, dt, t_k, ref_buf, hist_buf,
                       c_A, fin_f, hist_out);

  py::array_t<double> next_state({NX});
  py::array_t<double> next_hist({5});
  std::memcpy(next_state.mutable_data(), state_buf, sizeof(state_buf));
  std::memcpy(next_hist.mutable_data(), hist_out, sizeof(hist_out));
  return py::make_tuple(next_state, next_hist);
}

PYBIND11_MODULE(fish_dynamics, m) {
  m.doc() = "pybind11 wrapper around the fish RK4 dynamics step";
  m.def("step", &py_step, py::arg("state"), py::arg("body_params"),
        py::arg("fin_params"), py::arg("dt"), py::arg("t_k"),
        py::arg("action_ref"), py::arg("hist"), py::arg("c_A"),
        py::arg("fin_f"));
}
