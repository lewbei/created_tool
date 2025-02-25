This tool is to help convert any references into IJORAS references. 

The prompt output is the output from the Jupyter Lab is in plain text
The version 1 does not use .json. 
Version 2 and above required journals.json

The HTML output is the output from the Jupyter Lab is in the format.
All version required journals.json

The based_doi does not required json because it called the DOI API to check the DOI. 
This should be the efficient method to convert it to correct format.
It is possible the DOI does not has the correct page number, so you still need to check the reference if it is only shown single page instead of double page.


I just notice the based_doi has a problem whereby the doi maybe obtained other information which resutls into a different reference. Please dont use it.


