# Ficino Semantic Analysis Graph Descriptions

# Introduction

This file serves as a means of instructing the user of this project on how to 
interpret the scatterplots present in this directory.

## File Name

The file name contains the model that was run (skip gram, glove, or watson), and the text that it was run on. For
example, b1p1 means book 1 part 1. 

## 2D Scatter Plots

In the graphs, the x and y axis do not serve as any particular dimensions, they are merely the two dimensions that the 
multi-dimensional vectors were reduced into. This is why they are labeled with x and y. 

Since the scatter plots are displaying the results of the tsne dimension reduction of the embedded matrix understanding
of the text, this implies that the closer the words are (or rather their nodes), the more related they are. 

Size indicates the frequency of occurence of a particular word, and color represents the word's membership in a 
particular keyword cateogory. 

## Colors

###### Key:

1. Metaphysics - yellow
2. Theology - pink
3. Psychology - purple
4. Ethics - turquoise
5. Ontology - red
6. Figures & Influence - orange