/*
 * 当前Stage1部署使用的TensorRT IPluginV3掩码/选择CUDA实现。
 *
 * TensorRT 10.13在包含布尔常量、Cast和Where的融合图上可能出现Myelin内部编译失败。
 * 本插件给这些操作提供独立CUDA内核，避免依赖该融合实现；不替换Transformer、GRU
 * 权重或DDIM。插件API不支持BOOL，所以部署专用图将布尔值编码为INT32的0/1，原
 * Stage1采样接口与原始ONNX仍使用bool。插件支持FP32、INT32、INT64的线性张量。
 *
 * Where按广播坐标读取条件，只复制被选择分支的原始位，不用掩码乘加；因此未选择
 * 分支的NaN/Inf不会污染结果，负无穷注意力遮罩和符号零也保留。比较和逻辑运算
 * 输出0/1 INT32；Cast按输入实际类型转换，转bool时使用非零判定再输出0/1。
 * 所有内核使用TensorRT传入的CUDA stream，不同步、不分配设备内存、不调用Python。
 * 构建器用目标GPU架构及匹配的TensorRT头文件/实际运行库编译，随engine发布.so。
 * 加载端先核验.so的SHA256并显式注册creator，再反序列化engine；不依赖系统插件目录。
 * 本源码只支持固定Stage1图用到的九种操作；不提供旧GENMO网络支持或推理验收。
 */

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

// CUDA 13生成主机启动桩时，匿名命名空间与NvInfer头文件中的匿名空间会发生歧义。
namespace bumi_stage1_mask {
using namespace nvinfer1;
constexpr char kName[] = "BumiStage1Mask";
constexpr char kVersion[] = "1";
constexpr char kNamespace[] = "genmo_stage1";
constexpr int kMaxRank = 8;
enum Operation : int32_t { SELECT = 1, CAST = 2, CAST_BOOL = 3,
    EQUAL = 4, LESS = 5, GREATER = 6, AND = 7, OR = 8, NOT = 9 };

bool supported(DataType type) noexcept {
    return type == DataType::kFLOAT || type == DataType::kINT32 || type == DataType::kINT64;
}
int inputCount(int op) noexcept {
    return op == SELECT ? 3 : (op == CAST || op == CAST_BOOL || op == NOT ? 1 : 2);
}
struct KernelArgs {
    int32_t op{}, outputType{}, inputType{}, rank{};
    int64_t count{}, dimensions[kMaxRank]{}, strides[3][kMaxRank]{};
    void const* input[3]{};
    void* output{};
};

template <typename T>
__device__ bool compare(T left, T right, int op) {
    return op == EQUAL ? left == right : (op == LESS ? left < right : left > right);
}
template <typename T>
__device__ void convert(T value, KernelArgs const& args, int64_t index) {
    if (args.op == CAST_BOOL) {
        static_cast<int32_t*>(args.output)[index] = value != T(0);
    } else if (args.outputType == static_cast<int>(DataType::kFLOAT)) {
        static_cast<float*>(args.output)[index] = static_cast<float>(value);
    } else if (args.outputType == static_cast<int>(DataType::kINT32)) {
        static_cast<int32_t*>(args.output)[index] = static_cast<int32_t>(value);
    } else {
        static_cast<int64_t*>(args.output)[index] = static_cast<int64_t>(value);
    }
}

__global__ void maskKernel(KernelArgs args) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= args.count) return;
    int64_t offsets[3]{};
    int64_t remaining = index;
    for (int axis = args.rank - 1; axis >= 0; --axis) {
        int64_t coordinate = remaining % args.dimensions[axis];
        remaining /= args.dimensions[axis];
        for (int input = 0; input < 3; ++input) {
            offsets[input] += coordinate * args.strides[input][axis];
        }
    }
    if (args.op == SELECT) {
        int branch = static_cast<int32_t const*>(args.input[0])[offsets[0]] != 0 ? 1 : 2;
        // 逐位复制选中的候选值，不进行浮点乘加或读取另一分支的值。
        if (args.outputType == static_cast<int>(DataType::kINT64)) {
            static_cast<uint64_t*>(args.output)[index] =
                static_cast<uint64_t const*>(args.input[branch])[offsets[branch]];
        } else {
            static_cast<uint32_t*>(args.output)[index] =
                static_cast<uint32_t const*>(args.input[branch])[offsets[branch]];
        }
    } else if (args.op == CAST || args.op == CAST_BOOL) {
        if (args.inputType == static_cast<int>(DataType::kFLOAT)) {
            convert(static_cast<float const*>(args.input[0])[offsets[0]], args, index);
        } else if (args.inputType == static_cast<int>(DataType::kINT32)) {
            convert(static_cast<int32_t const*>(args.input[0])[offsets[0]], args, index);
        } else {
            convert(static_cast<int64_t const*>(args.input[0])[offsets[0]], args, index);
        }
    } else {
        bool result{};
        if (args.op == NOT || args.op == AND || args.op == OR) {
            bool left = static_cast<int32_t const*>(args.input[0])[offsets[0]] != 0;
            if (args.op == NOT) result = !left;
            else {
                bool right = static_cast<int32_t const*>(args.input[1])[offsets[1]] != 0;
                result = args.op == AND ? left && right : left || right;
            }
        } else if (args.inputType == static_cast<int>(DataType::kFLOAT)) {
            result = compare(static_cast<float const*>(args.input[0])[offsets[0]],
                static_cast<float const*>(args.input[1])[offsets[1]], args.op);
        } else if (args.inputType == static_cast<int>(DataType::kINT32)) {
            result = compare(static_cast<int32_t const*>(args.input[0])[offsets[0]],
                static_cast<int32_t const*>(args.input[1])[offsets[1]], args.op);
        } else {
            result = compare(static_cast<int64_t const*>(args.input[0])[offsets[0]],
                static_cast<int64_t const*>(args.input[1])[offsets[1]], args.op);
        }
        static_cast<int32_t*>(args.output)[index] = result;
    }
}

class MaskPlugin final : public IPluginV3, public IPluginV3OneCore,
    public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    MaskPlugin(int32_t op, int32_t outputType) : mOp(op), mOutputType(outputType) {}
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return nullptr;
    }
    IPluginV3* clone() noexcept override {
        return new (std::nothrow) MaskPlugin(mOp, mOutputType);
    }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getOutputDataTypes(DataType* out, int32_t nOut,
        DataType const* in, int32_t nIn) const noexcept override {
        if (nOut != 1 || nIn != inputCount(mOp)) return -1;
        auto type = static_cast<DataType>(mOutputType);
        if (!supported(type)) return -1;
        out[0] = type;
        return 0;
    }
    int32_t getOutputShapes(DimsExprs const* in, int32_t nIn, DimsExprs const*,
        int32_t nShapeIn, DimsExprs* out, int32_t nOut, IExprBuilder& builder) noexcept override {
        if (nShapeIn != 0 || nOut != 1 || nIn != inputCount(mOp)) return -1;
        int rank = 0;
        for (int i = 0; i < nIn; ++i) rank = std::max(rank, in[i].nbDims);
        if (rank > kMaxRank) return -1;
        out[0].nbDims = rank;
        for (int axis = 0; axis < rank; ++axis) {
            auto* extent = builder.constant(1);
            for (int i = 0; i < nIn; ++i) {
                int localAxis = axis - (rank - in[i].nbDims);
                if (localAxis >= 0) {
                    extent = builder.operation(DimensionOperation::kMAX, *extent, *in[i].d[localAxis]);
                    if (!extent) return -1;
                }
            }
            out[0].d[axis] = extent;
        }
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io,
        int32_t nIn, int32_t nOut) noexcept override {
        if (nIn != inputCount(mOp) || nOut != 1 || pos < 0 || pos > nIn
            || io[pos].desc.format != TensorFormat::kLINEAR) return false;
        DataType type = io[pos].desc.type;
        if (pos == nIn) return type == static_cast<DataType>(mOutputType);
        if (!supported(type)) return false;
        if (mOp == SELECT) {
            if (pos == 0) return type == DataType::kINT32;
            return type == static_cast<DataType>(mOutputType);
        }
        if (mOp == AND || mOp == OR || mOp == NOT) return type == DataType::kINT32;
        if (pos == 1) return type == io[0].desc.type;
        return true;
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nIn,
        DynamicPluginTensorDesc const* out, int32_t nOut) noexcept override {
        if (nIn != inputCount(mOp) || nOut != 1) return -1;
        PluginTensorDesc descriptions[3]{};
        for (int i = 0; i < nIn; ++i) descriptions[i] = in[i].desc;
        KernelArgs args{};
        return makeArgs(descriptions, out[0].desc, nIn, args);
    }
    int32_t onShapeChange(PluginTensorDesc const* in, int32_t nIn,
        PluginTensorDesc const* out, int32_t nOut) noexcept override {
        if (nIn != inputCount(mOp) || nOut != 1) return -1;
        KernelArgs args{};
        return makeArgs(in, out[0], nIn, args);
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const* out,
        void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override {
        KernelArgs args{};
        int count = inputCount(mOp);
        if (makeArgs(in, out[0], count, args) != 0) return -1;
        for (int i = 0; i < count; ++i) args.input[i] = inputs[i];
        args.output = outputs[0];
        maskKernel<<<static_cast<unsigned>((args.count + 255) / 256), 256, 0, stream>>>(args);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        mFields[0] = PluginField{"operation", &mOp, PluginFieldType::kINT32, 1};
        mFields[1] = PluginField{"output_type", &mOutputType, PluginFieldType::kINT32, 1};
        mCollection.nbFields = 2;
        mCollection.fields = mFields;
        return &mCollection;
    }
private:
    int32_t makeArgs(PluginTensorDesc const* in, PluginTensorDesc const& out,
        int count, KernelArgs& args) const noexcept {
        if (out.dims.nbDims < 0 || out.dims.nbDims > kMaxRank
            || out.type != static_cast<DataType>(mOutputType) || out.format != TensorFormat::kLINEAR) return -1;
        args.op = mOp;
        args.outputType = mOutputType;
        args.inputType = static_cast<int32_t>(in[0].type);
        args.rank = out.dims.nbDims;
        args.count = 1;
        for (int axis = 0; axis < args.rank; ++axis) {
            int64_t extent = out.dims.d[axis];
            if (extent <= 0 || args.count > std::numeric_limits<int64_t>::max() / extent) return -1;
            args.dimensions[axis] = extent;
            args.count *= extent;
        }
        for (int i = 0; i < count; ++i) {
            if (!supported(in[i].type) || in[i].format != TensorFormat::kLINEAR
                || in[i].dims.nbDims < 0 || in[i].dims.nbDims > args.rank) return -1;
            if (mOp == SELECT && ((i == 0 && in[i].type != DataType::kINT32)
                || (i > 0 && in[i].type != out.type))) return -1;
            if ((mOp == AND || mOp == OR || mOp == NOT) && in[i].type != DataType::kINT32) return -1;
            if ((mOp == EQUAL || mOp == LESS || mOp == GREATER) && in[i].type != in[0].type) return -1;
            int pad = args.rank - in[i].dims.nbDims;
            int64_t stride = 1;
            for (int axis = args.rank - 1; axis >= 0; --axis) {
                int64_t extent = axis < pad ? 1 : in[i].dims.d[axis - pad];
                if (extent != 1 && extent != args.dimensions[axis]) return -1;
                args.strides[i][axis] = extent == 1 ? 0 : stride;
                stride *= extent;
            }
        }
        return args.count <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()) * 256 ? 0 : -1;
    }
    int32_t mOp, mOutputType;
    PluginField mFields[2]{};
    PluginFieldCollection mCollection{};
};

class MaskCreator final : public IPluginCreatorV3One {
public:
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    PluginFieldCollection const* getFieldNames() noexcept override {
        return &mCollection;
    }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fields,
        TensorRTPhase) noexcept override {
        if (!fields) return nullptr;
        int32_t op = 0, type = -1;
        for (int i = 0; i < fields->nbFields; ++i) {
            auto const& field = fields->fields[i];
            if (!field.name || !field.data || field.length != 1) return nullptr;
            int32_t value{};
            if (field.type == PluginFieldType::kINT32) value = *static_cast<int32_t const*>(field.data);
            else if (field.type == PluginFieldType::kINT64) value = static_cast<int32_t>(*static_cast<int64_t const*>(field.data));
            else return nullptr;
            if (std::strcmp(field.name, "operation") == 0) op = value;
            else if (std::strcmp(field.name, "output_type") == 0) type = value;
            else return nullptr;
        }
        if (op < SELECT || op > NOT || !supported(static_cast<DataType>(type))) return nullptr;
        if (op != SELECT && op != CAST && type != static_cast<int32_t>(DataType::kINT32)) return nullptr;
        return new (std::nothrow) MaskPlugin(op, type);
    }
private:
    PluginField mFields[2]{{"operation", nullptr, PluginFieldType::kINT32, 1},
        {"output_type", nullptr, PluginFieldType::kINT32, 1}};
    PluginFieldCollection mCollection{2, mFields};
};
MaskCreator creator;
} // namespace bumi_stage1_mask

extern "C" bool bumi_stage1_register_plugins() noexcept {
    auto* registry = getPluginRegistry();
    return registry && registry->registerCreator(bumi_stage1_mask::creator, bumi_stage1_mask::kNamespace);
}
