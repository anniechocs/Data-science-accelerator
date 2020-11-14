# Data-science-accelerator
Project as part of my work at ONS to build an alternative commodity classification using NLP

## Installation

This project has a `setup.py` file which allows it to be installed with pip.

To do that navigate to the project directory and call:

```
pip install -e .
```

Here `-e` dynamically links the installed package to the source files,
so updates to the source code will be reflected when the module is imported.

## Running tests

Options:

* From an IDE
* From the command line with `pytest`
* From the command line with `python setup.py test`

# Data sources for Accerator project

## 1. CPA Data

"Statistical Classification of Products by Activity in the European Union, Version 2.1"  
[Downloaded from Eurostat RAMON database here](https://ec.europa.eu/eurostat/ramon/nomenclatures/index.cfm?TargetUrl=LST_CLS_DLD&StrNom=CPA_2_1&StrLanguageCode=EN&StrLayoutCode=HIERARCHIC)


## 2. CN Data 2020
"Combined Nomenclature, 2020"  
[Downloaded from Eurostat RAMON database here](https://ec.europa.eu/eurostat/ramon/nomenclatures/index.cfm?TargetUrl=LST_CLS_DLD&StrNom=CN_2020&StrLanguageCode=EN&StrLayoutCode=HIERARCHIC) 

## 3. COICOP Data 2018
"Classification of Individual Consumption by Purpose, 2018 version (COICOP 2018) "  
[Download from Eurostat RAMON database here](https://ec.europa.eu/eurostat/ramon/nomenclatures/index.cfm?TargetUrl=LST_CLS_DLD&StrNom=COICOP_2018&StrLanguageCode=EN&StrLayoutCode=HIERARCHIC)

## 4. SITC Data
"Commodity Indexes for the Standard International Trade Classification, Revision 3"
[Downloaded from UN site here - text file](https://unstats.un.org/unsd/tradekb/Knowledgebase/50096/Commodity-Indexes-for-the-Standard-International-Trade-Classification-Revision-3)

"Standard International Trade Classification, Revision 3 (1988)"
[Different version of Rev 3, with hierarchies](https://ec.europa.eu/eurostat/ramon/nomenclatures/index.cfm?TargetUrl=LST_CLS_DLD&StrNom=SITC_REV3&StrLanguageCode=EN&StrLayoutCode=HIERARCHIC)

Version 4 is not avaiable except as a pdf, but the conversion from 3 to 4 can be found

## 5. BEC codes
"Classification by Broad Economic Categories Defined in terms of the Harmonized Commodity Description and Coding System (2012) and the Central Product Classification, 2.1 (BEC Rev. 5, 2016)"  
[Downloaded from Eurostat RAMON database here](https://ec.europa.eu/eurostat/ramon/nomenclatures/index.cfm?TargetUrl=LST_NOM&StrLanguageCode=EN&IntFamilyCode=)
