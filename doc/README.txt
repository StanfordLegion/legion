README.txt

You probably just want to do "./crossIndex.bash"

glossary/html
contains a set of files, each file defines a glossary term, file names are exact (no extensions, no spaces in class names)

design_patterns/html
contains a set of files, each file contains a design pattern page, file names are full sentences

crossIndex.bash
script to automatically cross idnex the glossary and design patterns, results go into
publish/glossary
publish/design_patterns
publish/index.html

corssIndexGlossaryTerms.bash
this script is called by crossIndex.bash

createIndex.bash
this creates publish/index.html after the other scripts have run

