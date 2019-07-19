cc=arm-linux-g++ -std=c++11 -w -O3 -I./ `pkg-config --cflags --libs opencv`

LIBS=-L/opt/sgks/arm-linux/arm-buildroot-linux-gnueabi/sysroot/usr/lib  -lOpenCL

output:
	$(cc) $(LIBS)  host.cpp -o output


.PHONT:clean
clean:
	rm -rf *.o  *.a output