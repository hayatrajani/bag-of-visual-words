add_library(serialization serialization.cpp)
set_target_properties(serialization PROPERTIES PREFIX "")
target_link_libraries(serialization INTERFACE ${OpenCV_LIBS})

add_library(dataset dataset.cpp)
set_target_properties(dataset PROPERTIES PREFIX "")
target_link_libraries(dataset PRIVATE serialization dictionary PUBLIC histogram ${OpenCV_LIBS})

install(TARGETS serialization dataset DESTINATION lib)

