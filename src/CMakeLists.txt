get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_tool(dialectTest
dialectTest.cpp
)

target_link_libraries(dialectTest
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  )
