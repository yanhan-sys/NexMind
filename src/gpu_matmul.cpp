#include "gpu_matmul.h"

#ifdef _WIN32

#include <d3d11.h>
#include <d3dcompiler.h>
#include <windows.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

namespace nexmind {
namespace {

constexpr char kMatmulShader[] = R"hlsl(
StructuredBuffer<double> A : register(t0);
StructuredBuffer<double> B : register(t1);
RWStructuredBuffer<double> C : register(u0);

cbuffer MatmulParams : register(b0) {
    uint rows;
    uint inner;
    uint columns;
    uint reserved;
};

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= columns || id.y >= rows) {
        return;
    }

    double sum = 0.0;
    for (uint k = 0; k < inner; ++k) {
        sum += A[id.y * inner + k] * B[k * columns + id.x];
    }
    C[id.y * columns + id.x] = sum;
}
)hlsl";

struct MatmulParams {
    UINT rows;
    UINT inner;
    UINT columns;
    UINT reserved;
};

struct GpuState {
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    ID3D11ComputeShader* shader = nullptr;
    ID3D11Buffer* params = nullptr;
    bool available = false;

    ~GpuState() {
        if (params) params->Release();
        if (shader) shader->Release();
        if (context) context->Release();
        if (device) device->Release();
    }
};

GpuState& state() {
    static GpuState value;
    static std::once_flag flag;
    std::call_once(flag, [] {
        auto& s = value;
        D3D_FEATURE_LEVEL feature_level = D3D_FEATURE_LEVEL_11_0;
        const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0};

        HRESULT hr = D3D11CreateDevice(nullptr,
                                       D3D_DRIVER_TYPE_HARDWARE,
                                       nullptr,
                                       0,
                                       levels,
                                       1,
                                       D3D11_SDK_VERSION,
                                       &s.device,
                                       &feature_level,
                                       &s.context);
        if (FAILED(hr)) {
            s.available = false;
            return;
        }

        D3D11_FEATURE_DATA_DOUBLES doubles{};
        hr = s.device->CheckFeatureSupport(D3D11_FEATURE_DOUBLES,
                                           &doubles,
                                           sizeof(doubles));
        if (FAILED(hr) || !doubles.DoublePrecisionFloatShaderOps) {
            s.available = false;
            return;
        }

        ID3DBlob* shader_blob = nullptr;
        ID3DBlob* error_blob = nullptr;
        hr = D3DCompile(kMatmulShader,
                        sizeof(kMatmulShader) - 1,
                        "nexmind_gpu_matmul.hlsl",
                        nullptr,
                        nullptr,
                        "main",
                        "cs_5_0",
                        D3DCOMPILE_OPTIMIZATION_LEVEL3,
                        0,
                        &shader_blob,
                        &error_blob);
        if (FAILED(hr)) {
            if (error_blob) error_blob->Release();
            if (shader_blob) shader_blob->Release();
            s.available = false;
            return;
        }

        hr = s.device->CreateComputeShader(shader_blob->GetBufferPointer(),
                                            shader_blob->GetBufferSize(),
                                            nullptr,
                                            &s.shader);
        shader_blob->Release();
        if (error_blob) error_blob->Release();
        if (FAILED(hr)) {
            s.available = false;
            return;
        }

        D3D11_BUFFER_DESC params_desc{};
        params_desc.ByteWidth = sizeof(MatmulParams);
        params_desc.Usage = D3D11_USAGE_DYNAMIC;
        params_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        params_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        hr = s.device->CreateBuffer(&params_desc, nullptr, &s.params);
        if (FAILED(hr)) {
            s.available = false;
            return;
        }

        s.available = true;
    });
    return value;
}

std::mutex& execution_mutex() {
    static std::mutex mutex;
    return mutex;
}

bool create_input_buffer(ID3D11Device* device,
                         const double* data,
                         std::size_t count,
                         ID3D11Buffer** buffer) {
    if (count > static_cast<std::size_t>(UINT_MAX) / sizeof(double)) {
        return false;
    }
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(count * sizeof(double));
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(double);

    D3D11_SUBRESOURCE_DATA initial{};
    initial.pSysMem = data;
    return SUCCEEDED(device->CreateBuffer(&desc, &initial, buffer));
}

bool create_output_buffer(ID3D11Device* device,
                          std::size_t count,
                          ID3D11Buffer** buffer) {
    if (count > static_cast<std::size_t>(UINT_MAX) / sizeof(double)) {
        return false;
    }
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(count * sizeof(double));
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(double);
    return SUCCEEDED(device->CreateBuffer(&desc, nullptr, buffer));
}

bool create_readback_buffer(ID3D11Device* device,
                            std::size_t count,
                            ID3D11Buffer** buffer) {
    if (count > static_cast<std::size_t>(UINT_MAX) / sizeof(double)) {
        return false;
    }
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(count * sizeof(double));
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(double);
    return SUCCEEDED(device->CreateBuffer(&desc, nullptr, buffer));
}

bool create_srv(ID3D11Device* device,
                ID3D11Buffer* buffer,
                ID3D11ShaderResourceView** view) {
    D3D11_BUFFER_DESC buffer_desc{};
    buffer->GetDesc(&buffer_desc);
    D3D11_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.BufferEx.FirstElement = 0;
    desc.BufferEx.NumElements = buffer_desc.ByteWidth / buffer_desc.StructureByteStride;
    return SUCCEEDED(device->CreateShaderResourceView(buffer, &desc, view));
}

bool create_uav(ID3D11Device* device,
                ID3D11Buffer* buffer,
                ID3D11UnorderedAccessView** view) {
    D3D11_BUFFER_DESC buffer_desc{};
    buffer->GetDesc(&buffer_desc);
    D3D11_UNORDERED_ACCESS_VIEW_DESC desc{};
    desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.Buffer.FirstElement = 0;
    desc.Buffer.NumElements = buffer_desc.ByteWidth / buffer_desc.StructureByteStride;
    return SUCCEEDED(device->CreateUnorderedAccessView(buffer, &desc, view));
}

bool use_gpu() {
    const char* value = std::getenv("NEXMIND_DEVICE");
    if (!value || std::string(value) == "auto") {
        return true;
    }
    return std::string(value) == "gpu";
}

} // namespace

bool gpu_matmul_available() {
    return use_gpu() && state().available;
}

bool gpu_matmul(const double* lhs,
                const double* rhs,
                double* result,
                std::size_t rows,
                std::size_t inner,
                std::size_t columns) {
    if (!use_gpu() || !state().available || !lhs || !rhs || !result ||
        rows == 0 || inner == 0 || columns == 0 ||
        rows > UINT_MAX || inner > UINT_MAX || columns > UINT_MAX) {
        return false;
    }

    std::lock_guard<std::mutex> lock(execution_mutex());
    auto& s = state();
    ID3D11Buffer* a = nullptr;
    ID3D11Buffer* b = nullptr;
    ID3D11Buffer* c = nullptr;
    ID3D11Buffer* readback = nullptr;
    ID3D11ShaderResourceView* a_view = nullptr;
    ID3D11ShaderResourceView* b_view = nullptr;
    ID3D11UnorderedAccessView* c_view = nullptr;

    const bool ok =
        create_input_buffer(s.device, lhs, rows * inner, &a) &&
        create_input_buffer(s.device, rhs, inner * columns, &b) &&
        create_output_buffer(s.device, rows * columns, &c) &&
        create_readback_buffer(s.device, rows * columns, &readback) &&
        create_srv(s.device, a, &a_view) &&
        create_srv(s.device, b, &b_view) &&
        create_uav(s.device, c, &c_view);
    if (!ok) {
        if (c_view) c_view->Release();
        if (b_view) b_view->Release();
        if (a_view) a_view->Release();
        if (readback) readback->Release();
        if (c) c->Release();
        if (b) b->Release();
        if (a) a->Release();
        return false;
    }

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(s.context->Map(s.params, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        c_view->Release();
        b_view->Release();
        a_view->Release();
        readback->Release();
        c->Release();
        b->Release();
        a->Release();
        return false;
    }
    auto* params = static_cast<MatmulParams*>(mapped.pData);
    params->rows = static_cast<UINT>(rows);
    params->inner = static_cast<UINT>(inner);
    params->columns = static_cast<UINT>(columns);
    params->reserved = 0;
    s.context->Unmap(s.params, 0);

    ID3D11ShaderResourceView* srvs[] = {a_view, b_view};
    ID3D11UnorderedAccessView* uavs[] = {c_view};
    ID3D11Buffer* constant_buffers[] = {s.params};
    s.context->CSSetShader(s.shader, nullptr, 0);
    s.context->CSSetShaderResources(0, 2, srvs);
    s.context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    s.context->CSSetConstantBuffers(0, 1, constant_buffers);
    s.context->Dispatch(static_cast<UINT>((columns + 7) / 8),
                        static_cast<UINT>((rows + 7) / 8),
                        1);

    s.context->CopyResource(readback, c);
    D3D11_MAPPED_SUBRESOURCE output{};
    const HRESULT map_hr = s.context->Map(readback, 0, D3D11_MAP_READ, 0, &output);
    if (SUCCEEDED(map_hr)) {
        const std::size_t count = rows * columns;
        std::copy_n(static_cast<const double*>(output.pData), count, result);
        s.context->Unmap(readback, 0);
    }

    ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
    ID3D11UnorderedAccessView* null_uavs[] = {nullptr};
    s.context->CSSetShaderResources(0, 2, null_srvs);
    s.context->CSSetUnorderedAccessViews(0, 1, null_uavs, nullptr);
    s.context->CSSetShader(nullptr, nullptr, 0);

    c_view->Release();
    b_view->Release();
    a_view->Release();
    readback->Release();
    c->Release();
    b->Release();
    a->Release();
    return SUCCEEDED(map_hr);
}

} // namespace nexmind

#else

namespace nexmind {

bool gpu_matmul_available() {
    return false;
}

bool gpu_matmul(const double*, const double*, double*, std::size_t, std::size_t, std::size_t) {
    return false;
}

} // namespace nexmind

#endif
