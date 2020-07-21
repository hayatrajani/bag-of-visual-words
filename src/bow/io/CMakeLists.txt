add_library(dataset dataset.cpp)
set_target_properties(dataset PROPERTIES PREFIX "")
target_link_libraries(dataset PRIVATE dictionary PUBLIC descriptor histogram ${OpenCV_LIBS})

install(TARGETS dataset DESTINATION lib)
