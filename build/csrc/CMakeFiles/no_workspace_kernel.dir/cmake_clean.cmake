file(REMOVE_RECURSE
  "../lib/libno_workspace_kernel.a"
  "../lib/libno_workspace_kernel.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/no_workspace_kernel.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
