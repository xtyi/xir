from xir import DSLParseError, kernel


def test_restricted_kernel_builds_schedule_ast():
    @kernel
    def gemm(A, B):
        smem = shared((128, 64), dtype="bf16", stages=3)
        pipe = pipeline(stages=3)
        with role(warps=[0]):
            for stage in stages(pipe):
                async_copy(A, smem, barrier="ready")
        with role(warps=[1]):
            for stage in stages(pipe):
                wait("ready", stage)
                sqmma("acc", smem, B)

    tree = gemm.build().to_dict()
    assert tree["kind"] == "Kernel"
    assert tree["children"][0]["kind"] == "BufferDecl"
    assert tree["children"][2]["kind"] == "RoleRegion"


def test_unknown_operation_has_location():
    @kernel
    def bad(A):
        nope(A)

    try:
        bad.build()
    except DSLParseError as error:
        assert error.code == "XIR001"
        assert "unknown DSL operation" in str(error)
    else:
        raise AssertionError("expected DSLParseError")
