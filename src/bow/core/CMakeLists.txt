add_library(descriptor descriptor.cpp)
set_target_properties(descriptor PROPERTIES PREFIX "")
target_link_libraries(descriptor PUBLIC ${OpenCV_LIBS})

add_library(dictionary dictionary.cpp)
set_target_properties(dictionary PROPERTIES PREFIX "")
target_link_libraries(dictionary PRIVATE algorithms INTERFACE descriptor PUBLIC ${OpenCV_LIBS})

add_library(histogram histogram.cpp)
set_target_properties(histogram PROPERTIES PREFIX "")
target_link_libraries(histogram PRIVATE algorithms PUBLIC dictionary ${OpenCV_LIBS})

install(TARGETS descriptor dictionary histogram DESTINATION lib)
