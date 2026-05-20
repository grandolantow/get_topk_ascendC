add_library(ascendc_runtime_obj OBJECT IMPORTED)
set_target_properties(ascendc_runtime_obj PROPERTIES
    IMPORTED_OBJECTS "/home/s886374/kvoffload/ascend-kernel/build/ascendc_runtime.cpp.o;/home/s886374/kvoffload/ascend-kernel/build/aicpu_rt.cpp.o;/home/s886374/kvoffload/ascend-kernel/build/ascendc_elf_tool.c.o"
)
