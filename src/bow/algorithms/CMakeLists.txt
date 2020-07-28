add_library(algorithms algorithms.cpp)
set_target_properties(algorithms PROPERTIES PREFIX "")
target_link_libraries(algorithms PUBLIC ${OpenCV_LIBS})

install(TARGETS algorithms DESTINATION lib)
