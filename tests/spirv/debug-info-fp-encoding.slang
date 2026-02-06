//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry computeMain -stage compute -g2 -emit-spirv-directly -O0

// Test that DebugTypeBasic includes the FPEncoding operand for
// non-IEEE floating-point types (BFloat16, FloatE4M3, FloatE5M2).
// These types can only be used with CoopMat (not CoopVec), so we
// wrap each in a struct to force DebugTypeMember emission which
// triggers emitDebugType for the element type.

using namespace linalg;

ByteAddressBuffer inputBuffer;
RWStructuredBuffer<float> outputBuffer;

typealias BF16Mat = CoopMat<BFloat16, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse::MatrixA>;
typealias E4M3Mat = CoopMat<FloatE4M3, MemoryScope.Subgroup, 16, 32, CoopMatMatrixUse::MatrixA>;
typealias E5M2Mat = CoopMat<FloatE5M2, MemoryScope.Subgroup, 16, 32, CoopMatMatrixUse::MatrixA>;

struct BF16Holder
{
    BF16Mat mat;
}

struct E4M3Holder
{
    E4M3Mat mat;
}

struct E5M2Holder
{
    E5M2Mat mat;
}

void processBF16(inout BF16Holder h)
{
    h.mat = h.mat;
}

void processE4M3(inout E4M3Holder h)
{
    h.mat = h.mat;
}

void processE5M2(inout E5M2Holder h)
{
    h.mat = h.mat;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void computeMain()
{
    BF16Holder bh;
    bh.mat = BF16Mat.Load<CoopMatMatrixLayout::RowMajor>(inputBuffer, 0, 16);
    processBF16(bh);

    E4M3Holder eh4;
    eh4.mat = E4M3Mat.Load<CoopMatMatrixLayout::RowMajor>(inputBuffer, 0, 16);
    processE4M3(eh4);

    E5M2Holder eh5;
    eh5.mat = E5M2Mat.Load<CoopMatMatrixLayout::RowMajor>(inputBuffer, 0, 16);
    processE5M2(eh5);

    outputBuffer[0] = 0.0;
}

// Verify the import string uses 110.
// CHECK: OpExtInstImport "NonSemantic.Shader.DebugInfo.110"

// Verify DebugTypeBasic with FPEncoding for BFloat16 (5 operands, FPEncoding=BFloat16KHR=0).
// CHECK: OpExtInst %void {{%[0-9]+}} 2 {{.*}} %uint_16 %uint_3 {{.*}} %uint_0

// Verify DebugTypeBasic with FPEncoding for FloatE4M3 (5 operands, FPEncoding=Float8E4M3EXT=4214).
// CHECK: OpExtInst %void {{%[0-9]+}} 2 {{.*}} %uint_8 %uint_3 {{.*}} %uint_4214

// Verify DebugTypeBasic with FPEncoding for FloatE5M2 (5 operands, FPEncoding=Float8E5M2EXT=4215).
// CHECK: OpExtInst %void {{%[0-9]+}} 2 {{.*}} %uint_8 %uint_3 {{.*}} %uint_4215
