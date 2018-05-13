CXXFLAGS:= $(shell pkg-config --cflags opencv)

LDLIBS:= $(shell pkg-config --libs opencv)

lane_detection_cpp:main.o pipline.o caliberation.o line_fit.o
	g++ -o lane_detection_cpp main.o pipline.o caliberation.o line_fit.o -g -O2 -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib  -lopencv_calib3d -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -larmadillo -llapack -lblas

main.o:main.cpp caliberation.h pipline.h
	g++ -c main.cpp $(CXXFLAGS)

line_fit.o: line_fit.cpp line_fit.h
	g++ -c line_fit.cpp $(CXXFLAGS)

pipline.o:pipline.cpp pipline.h
	g++ -c pipline.cpp $(CXXFLAGS)
caliberation.o: caliberation.cpp caliberation.h
	g++ -c  caliberation.cpp $(CXXFLAGS)



clean:
	rm -f *.o
