NVCC = nvcc
CXX = g++
NVCCFLAGS = -std=c++11
CXXFLAGS = -std=c++11

all: addition

logger.o: logger.cpp logger.h
	$(CXX) $(CXXFLAGS) -c logger.cpp -o logger.o

addition.o: addition.cu logger.h
	$(NVCC) $(NVCCFLAGS) -c addition.cu -o addition.o

addition: addition.o logger.o
	$(NVCC) $(NVCCFLAGS) addition.o logger.o -o addition

clean:
	rm -f *.o addition
