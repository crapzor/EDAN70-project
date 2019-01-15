# EDAN70-project

This project was to create an entity linker using named entity recognition for the course EDAN70 Project in Language Technology.

In order to run the program, run the programs in the following order:
          - converter (converts the CoNLL files from IOBv1 to IOBv2 formatting)
          - chunker (creates the model. takes the converted CoNLL files and returns files with the predicted chunk-tags in a new column)
          - mention_extractor (baseline linking extracted from wikipedia corpora)
          - file_uniformer (formats files to a uniformed standard to use in neleval_prep)
          - neleval_prep (converts files to TAC-2014 format for evaluation with neleval)
