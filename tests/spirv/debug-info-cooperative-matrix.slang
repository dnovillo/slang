//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry computeMain -stage compute -g2 -emit-spirv-directly -O0

// Test that DebugTypeCooperativeMatrixKHR (instruction 111) is emitted
// when using CoopMat with debug info enabled.

using namespace linalg;

typealias CoopMatType = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;

StructuredBuffer<float> inputBuffer;
RWStructuredBuffer<float> outputBuffer;

struct MatHolder
{
    CoopMatType mat;
}

void processMat(inout MatHolder h)
{
    h.mat = h.mat * 2.0;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void computeMain()
{
    MatHolder h;
    h.mat = CoopMatType.Load<CoopMatMatrixLayout::RowMajor>(inputBuffer, 0, 16);
    processMat(h);
    h.mat.Store<CoopMatMatrixLayout::RowMajor>(outputBuffer, 0, 16);
}

// Verify the import string uses 110.
// CHECK: OpExtInstImport "NonSemantic.Shader.DebugInfo.110"

// Verify DebugTypeBasic for float (instruction 2).
// CHECK: [[DBG_FLOAT:%[0-9]+]] = OpExtInst %void {{%[0-9]+}} 2 {{.*}} %uint_32 %uint_3

// Verify DebugTypeCooperativeMatrixKHR (instruction 111).
// Operands: componentType, scope(Subgroup=3), rows(16), columns(16), use(MatrixAccumulator=2).
// CHECK: OpExtInst %void {{%[0-9]+}} 111 [[DBG_FLOAT]] %uint_3 %uint_16 %uint_16 %uint_2
