#include "ffnbrick.h"
#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FFNBrick optimized native C++/CUDA inference backend";

    m.def("micro_virtual_forward_packed", &ffnbrick::micro_virtual_forward_packed,
          pybind11::arg("x"), pybind11::arg("gate_up_w"),
          pybind11::arg("down_w"), pybind11::arg("fused_cuda") = true);

    m.def("micro_virtual_refine_packed", &ffnbrick::micro_virtual_refine_packed,
          pybind11::arg("x"), pybind11::arg("gate_up_w"),
          pybind11::arg("down_w"), pybind11::arg("refinements"),
          pybind11::arg("fused_cuda") = true);

    m.def("state_aware_forward_packed", &ffnbrick::state_aware_forward_packed,
          pybind11::arg("x"), pybind11::arg("esa_update"),
          pybind11::arg("previous_esa"), pybind11::arg("previous_state"),
          pybind11::arg("params"), pybind11::arg("fused_cuda") = true);

    m.def("virtual_state_aware_forward_packed", &ffnbrick::virtual_state_aware_forward_packed,
          pybind11::arg("x"), pybind11::arg("esa_update"),
          pybind11::arg("previous_esa"), pybind11::arg("previous_state"),
          pybind11::arg("params"), pybind11::arg("virtual_params"),
          pybind11::arg("refinements"), pybind11::arg("rms_eps") = 1e-6,
          pybind11::arg("fused_cuda") = true);
}
