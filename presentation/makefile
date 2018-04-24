
.PHONY: all build clean

all:
	node asciidoctor-revealjs.js
	xdg-open presentation.html

install:
	npm i --save asciidoctor-reveal.js

clean:
	rm -rf node_modules
	rm -f package-lock.json

