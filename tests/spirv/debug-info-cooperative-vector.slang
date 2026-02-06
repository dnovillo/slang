//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry computeMain -stage compute -g2 -emit-spirv-directly -O0

// Test that DebugTypeCooperativeVectorNV (instruction 109) is emitted
// when using CoopVec with debug info enabled.

RWStructuredBuffer<float> outputBuffer;

struct Holder
{
    CoopVec<float, 4> vec;
}

void processVec(inout Holder h)
{
    h.vec = h.vec + h.vec;
}

[numthreads(1, 1, 1)]
void computeMain()
{
    Holder h;
    h.vec = CoopVec<float, 4>(1.0, 2.0, 3.0, 4.0);
    processVec(h);
    h.vec.store(outputBuffer, 0);
}

// Verify the import string uses 110.
// CHECK: OpExtInstImport "NonSemantic.Shader.DebugInfo.110"

// Verify DebugTypeBasic for float (instruction 2).
// CHECK: [[DBG_FLOAT:%[0-9]+]] = OpExtInst %void {{%[0-9]+}} 2 {{.*}} %uint_32 %uint_3

// Verify DebugTypeCooperativeVectorNV (instruction 109).
// CHECK: OpExtInst %void {{%[0-9]+}} 109 [[DBG_FLOAT]] %uint_4
