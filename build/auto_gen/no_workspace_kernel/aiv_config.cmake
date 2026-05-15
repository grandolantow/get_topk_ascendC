set(MIX_SOURCES
)
set(AIV_SOURCES
    /home/s886374/kvoffload/ascend-kernel/build/auto_gen/no_workspace_kernel/auto_gen_cache_miss_topk.cpp
    /home/s886374/kvoffload/ascend-kernel/build/auto_gen/no_workspace_kernel/auto_gen_kernel_helloworld.cpp
)
set_source_files_properties(/home/s886374/kvoffload/ascend-kernel/build/auto_gen/no_workspace_kernel/auto_gen_cache_miss_topk.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_cache_miss_topk_kernel=cache_miss_topk_0;ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/home/s886374/kvoffload/ascend-kernel/build/auto_gen/no_workspace_kernel/auto_gen_kernel_helloworld.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_helloworld_kernel=helloworld_1;ONE_CORE_DUMP_SIZE=1048576"
)
