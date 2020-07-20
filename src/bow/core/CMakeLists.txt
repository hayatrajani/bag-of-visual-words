add_library(kmeans kmeans.cpp)
set_target_properties(kmeans PROPERTIES PREFIX "")
target_link_libraries(kmeans PUBLIC ${OpenCV_LIBS})

add_library(dictionary dictionary.cpp)
set_target_properties(dictionary PROPERTIES PREFIX "")
target_link_libraries(dictionary PRIVATE kmeans PUBLIC ${OpenCV_LIBS})

add_library(histogram histogram.cpp)
set_target_properties(histogram PROPERTIES PREFIX "")
target_link_libraries(histogram PUBLIC dictionary ${OpenCV_LIBS})

install(TARGETS kmeans dictionary histogram DESTINATION lib)
