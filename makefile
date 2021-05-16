CFLAGS=
fastblur: fastblur.o
        gcc $(CFLAGS) fastblur.o -o fastblur -lm


fastblur.o: fastblur.c
        gcc -c $(CFLAGS) fastblur.c -o fastblur.o 


clean:
        rm -f obj/* fastblur output.png

