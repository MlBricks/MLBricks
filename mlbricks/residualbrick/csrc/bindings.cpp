#include "residualbrick.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ResidualBrick native C++/CUDA backend";
    m.def(
        "residual_forward",
        &residualbrick::residual_forward,
        pybind11::arg("residual"),
        pybind11::arg("update"),
        pybind11::arg("update_ratio"),
        pybind11::arg("stream_ratio"),
        pybind11::arg("update_softness"),
        pybind11::arg("stream_softness"),
        pybind11::arg("eps"),
        pybind11::arg("fused_cuda") = true);
}
