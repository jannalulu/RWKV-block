#include <torch/extension.h>
#include <cuda_bf16.h>
using bf = __nv_bfloat16;

void cuda_forward(int B, int T, int H, float*_state, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa);

void forward(torch::Tensor &_state, torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, (float*)_state.data_ptr(), (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr());
}

void cuda_backward(int B, int T, int H, float*_state, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float*s, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da);

void backward(torch::Tensor &_state, torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, (float*)_state.data_ptr(), (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)dy.data_ptr(), 
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)dz.data_ptr(), (bf*)da.data_ptr());
}

TORCH_LIBRARY(state_wind_backstepping, m) {
    m.def("forward(Tensor _state, Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa) -> ()");
    m.def("backward(Tensor _state, Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da) -> ()");
}

TORCH_LIBRARY_IMPL(state_wind_backstepping, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}

// TORCH_LIBRARY(state_wind_backstepping, m) {
//     m.def("forward", forward);
//     m.def("backward", backward);
// }