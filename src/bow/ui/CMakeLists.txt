add_library(mat_serialization src/serialize.cpp)
set_target_properties(mat_serialization PROPERTIES PREFIX "")
target_link_libraries(mat_serialization INTERFACE ${OpenCV_LIBS})

add_library(convert_dataset src/convert_dataset.cpp)
set_target_properties(convert_dataset PROPERTIES PREFIX "")
target_link_libraries(convert_dataset PUBLIC ${OpenCV_LIBS} PRIVATE mat_serialization)

add_executable(test_serialization src/test_serialization.cpp)
target_link_libraries(test_serialization mat_serialization)

add_executable(main src/main.cpp)
target_link_libraries(main convert_dataset)

install(TARGETS bow
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
