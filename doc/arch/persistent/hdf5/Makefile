SHELL = /bin/sh

.PHONY: all clean

TARGET := io_proposal.tex

all: clean
#	cd bib && ./make.sh && cd ..
	latexmk -pdf $(TARGET)	

clean:
	latexmk -C $(TARGET)	
