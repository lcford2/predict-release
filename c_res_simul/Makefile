CC=gcc
CFLAGS=-Werror -Wall -g
DIRLOC=$${HOME}/predict-release/c_res_simul
LIBFLAGS=-I$(DIRLOC)/include -L$(DIRLOC)/lib -Wl,-rpath=$(DIRLOC)/lib -lqueue
LDFLAGS=-shared
OPTFLAGS=-Ofast

all: res_simul.o libres_simul.so

res_simul.o: src/res_simul.c
	 $(CC) $(CFLAGS) $(LIBFLAGS) $(OPTFLAGS) -c -fPIC $^ -o obj/$@

libres_simul.so: obj/res_simul.o
	$(CC) $(LDFLAGS) $(LIBFLAGS) $^ -o lib/$@
# mv res_simul.o obj
# mv $@ lib

test_queue: test_queue.c
	$(CC) $(CFLAGS) $(LIBFLAGS) $^ -o $@

clean:
	rm obj/res_simul.o lib/libres_simul.so
