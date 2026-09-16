#include "gpu_matmul.h"

#ifdef _WIN32

#include <d3d11.h>
#include <d3dcompiler.h>
#include <windows.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

namespace nexmind {
namespace {

// 使用 16x16 tile，让 GPU 线程组复用 A/B 数据，减少全局内存访问。
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

groupshared double tile_a[16][16];
groupshared double tile_b[16][16];

[numthreads(16, 16, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
    const uint row = group_id.y * 16 + group_thread_id.y;
    const uint column = group_id.x * 16 + group_thread_id.x;
    double sum = 0.0;

    const uint tile_count = (inner + 15) / 16;
    for (uint tile = 0; tile < tile_count; ++tile) {
        const uint a_column = tile * 16 + group_thread_id.x;
        const uint b_row = tile * 16 + group_thread_id.y;

        tile_a[group_thread_id.y][group_thread_id.x] =
            (row < rows && a_column < inner) ? A[row * inner + a_column] : 0.0;
        tile_b[group_thread_id.y][group_thread_id.x] =
            (b_row < inner && column < columns) ? B[b_row * columns + column] : 0.0;

        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < 16; ++k) {
            sum += tile_a[group_thread_id.y][k] * tile_b[k][group_thread_id.x];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < rows && column < columns) {
        C[row * columns + column] = sum;
    }
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
    ID3D11Buffer* a = nullptr;
    ID3D11Buffer* b = nullptr;
    ID3D11Buffer* c = nullptr;
    ID3D11Buffer* readback = nullptr;
    ID3D11ShaderResourceView* a_view = nullptr;
    ID3D11ShaderResourceView* b_view = nullptr;
    ID3D11UnorderedAccessView* c_view = nullptr;
    std::size_t a_count = 0;
    std::size_t b_count = 0;
    std::size_t c_count = 0;
    bool available = false;

    ~GpuState() {
        release_buffers();
        if (params) params->Release();
        if (shader) shader->Release();
        if (context) context->Release();
        if (device) device->Release();
    }

    void release_buffers() {
        if (c_view) c_view->Release();
        if (b_view) b_view->Release();
        if (a_view) a_view->Release();
        if (readback) readback->Release();
        if (c) c->Release();
        if (b) b->Release();
        if (a) a->Release();
        c_view = nullptr;
        b_view = nullptr;
        a_view = nullptr;
        readback = nullptr;
        c = nullptr;
        b = nullptr;
        a = nullptr;
        a_count = 0;
        b_count = 0;
        c_count = 0;
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

bool create_dynamic_input_buffer(ID3D11Device* device,
                                 std::size_t count,
                                 ID3D11Buffer** buffer) {
    if (count == 0 || count > static_cast<std::size_t>(UINT_MAX) / sizeof(double)) {
        return false;
    }
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(count * sizeof(double));
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(double);
    return SUCCEEDED(device->CreateBuffer(&desc, nullptr, buffer));
}

bool create_output_buffer(ID3D11Device* device,
                          std::size_t count,
                          ID3D11Buffer** buffer) {
    if (count == 0 || count > static_cast<std::size_t>(UINT_MAX) / sizeof(double)) {
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
    if (count == 0 || count > static_cast<std::size_t>(UINT_MAX) / sizeof(double)) {
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

bool ensure_buffers(GpuState& s,
                    std::size_t a_count,
                    std::size_t b_count,
                    std::size_t c_count) {
    if (s.a_count == a_count && s.b_count == b_count && s.c_count == c_count &&
        s.a && s.b && s.c && s.readback && s.a_view && s.b_view && s.c_view) {
        return true;
    }

    s.release_buffers();
    if (!create_dynamic_input_buffer(s.device, a_count, &s.a) ||
        !create_dynamic_input_buffer(s.device, b_count, &s.b) ||
        !create_output_buffer(s.device, c_count, &s.c) ||
        !create_readback_buffer(s.device, c_count, &s.readback) ||
        !create_srv(s.device, s.a, &s.a_view) ||
        !create_srv(s.device, s.b, &s.b_view) ||
        !create_uav(s.device, s.c, &s.c_view)) {
        s.release_buffers();
        return false;
    }

    s.a_count = a_count;
    s.b_count = b_count;
    s.c_count = c_count;
    return true;
}

bool upload_buffer(ID3D11DeviceContext* context,
                   ID3D11Buffer* buffer,
                   const double* data,
                   std::size_t count) {
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(context->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        return false;
    }
    std::memcpy(mapped.pData, data, count * sizeof(double));
    context->Unmap(buffer, 0);
    return true;
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

    if (rows > static_cast<std::size_t>(UINT_MAX) / inner ||
        inner > static_cast<std::size_t>(UINT_MAX) / columns ||
        rows > static_cast<std::size_t>(UINT_MAX) / columns) {
        return false;
    }

    std::lock_guard<std::mutex> lock(execution_mutex());
    auto& s = state();
    const std::size_t a_count = rows * inner;
    const std::size_t b_count = inner * columns;
    const std::size_t c_count = rows * columns;

    if (!ensure_buffers(s, a_count, b_count, c_count) ||
        !upload_buffer(s.context, s.a, lhs, a_count) ||
        !upload_buffer(s.context, s.b, rhs, b_count)) {
        return false;
    }

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(s.context->Map(s.params, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        return false;
    }
    auto* params = static_cast<MatmulParams*>(mapped.pData);
    params->rows = static_cast<UINT>(rows);
    params->inner = static_cast<UINT>(inner);
    params->columns = static_cast<UINT>(columns);
    params->reserved = 0;
    s.context->Unmap(s.params, 0);

    ID3D11ShaderResourceView* srvs[] = {s.a_view, s.b_view};
    ID3D11UnorderedAccessView* uavs[] = {s.c_view};
    ID3D11Buffer* constant_buffers[] = {s.params};
    s.context->CSSetShader(s.shader, nullptr, 0);
    s.context->CSSetShaderResources(0, 2, srvs);
    s.context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    s.context->CSSetConstantBuffers(0, 1, constant_buffers);
    s.context->Dispatch(static_cast<UINT>((columns + 15) / 16),
                        static_cast<UINT>((rows + 15) / 16),
                        1);

    s.context->CopyResource(s.readback, s.c);
    D3D11_MAPPED_SUBRESOURCE output{};
    const HRESULT map_hr = s.context->Map(s.readback, 0, D3D11_MAP_READ, 0, &output);
    if (SUCCEEDED(map_hr)) {
        std::copy_n(static_cast<const double*>(output.pData), c_count, result);
        s.context->Unmap(s.readback, 0);
    }

    ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
    ID3D11UnorderedAccessView* null_uavs[] = {nullptr};
    s.context->CSSetShaderResources(0, 2, null_srvs);
    s.context->CSSetUnorderedAccessViews(0, 1, null_uavs, nullptr);
    s.context->CSSetShader(nullptr, nullptr, 0);

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
