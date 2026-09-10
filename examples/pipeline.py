"""Protocol-only example: repeats the same two input tiles; no GPU execution."""
from xir import kernel


@kernel(block=(160, 1, 1))
def pipeline_example(a_ptr: "const_ptr[bf16]", b_ptr: "const_ptr[bf16]", iterations: "index"):
    a_storage = storage("Global", 16384, 128, external=a_ptr)
    b_storage = storage("Global", 16384, 128, external=b_ptr)
    sa_storage = storage("Shared", 32768, 128)
    sb_storage = storage("Shared", 32768, 128)
    a = buffer(a_storage, "bf16", (128, 64), (128, 2), 0, False)
    b = buffer(b_storage, "bf16", (64, 128), (256, 2), 0, False)
    sa = buffer(sa_storage, "bf16", (128, 64), (128, 2), 0, True)
    sb = buffer(sb_storage, "bf16", (64, 128), (256, 2), 0, True)
    producer = role([4])
    consumer = role([0, 1, 2, 3])
    pipe = pipeline(iterations, 2, producer, consumer, [sa, sb], [16384, 16384])
    acc = accumulator("f32", (128, 128), consumer, "mp31.sqmma.acc.128x128.f32")
    with concurrent_roles():
        with role_region(producer):
            for i in pipeline_iterations(pipe):
                write = acquire(pipe, i)
                copy_a = async_copy(a, leased(sa, write), "mp31.tme.load")
                copy_b = async_copy(b, leased(sb, write), "mp31.tme.load")
                publish(write, [copy_a, copy_b])
            drain(pipe)
        with role_region(consumer):
            accumulator_init(acc, 0.0)
            for i in pipeline_iterations(pipe):
                read = consume(pipe, i)
                mma = sqmma(acc, leased(sa, read), leased(sb, read), "MP31_128x128x64_F32BF16BF16_SS")
                await_event(mma)
                release(read)
            drain(pipe)
            fragment = accumulator_read(acc)


if __name__ == "__main__":
    import json
    print(json.dumps(pipeline_example.verify(), indent=2))
