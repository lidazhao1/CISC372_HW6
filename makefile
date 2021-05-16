CFLAGS=
cudaBlur: obj/cudaBlur.o
	nvcc $(CFLAGS) obj/cudaBlur.o -o cudaBlur -lm


obj/cudaBlur.o: cudaBlur.c
	nvcc -c $(CFLAGS) cudaBlur.c -o obj/cudaBlur.o 


clean:
	rm -f obj/* cudaBlur output.png

