CXXFLAGS =-std=c++11 \
		  -O3 $(shell pkg-config --cflags libjpeg) $(shell python3-config --cflags) -UNDEBUG
LDFLAGS = $(shell pkg-config --libs libjpeg) $(shell python3-config --ldflags)

CXX_SOURCES = _jpegpy.cc

LIB_SO = _jpegpy.so

$(LIB_SO) : $(CXX_SOURCES)
	$(CXX) $(CXX_SOURCES) -o $@ $(CXXFLAGS) $(LDFLAGS) --shared -fPIC

clean:
	rm -rf $(LIB_SO)
