This Java package performs cluster center initialization (CCIA) for K-means algorithms. The implementation is based on the following paper - 

Cluster Center Initialization Algorithm for K-mean Clustering, Shehroz S. Khan and Amir Ahmad, Pattern Recognition Letters, Volume 25, No. 11, pages 1293-1302, 2004 

Online available at - https://pdfs.semanticscholar.org/0288/181f90c5f85ba219ebc4beb7c759fd052408.pdf

Author: Shehroz S. Khan
Affiliation: University of Toronto, Canada

*About*: The program generates fixed cluster centers that are written to a separate file. These centers are then used by K-means clustering and the data object membership is written in another file. 

This distribution contains two java files:

1. KmeansClustering.java -- This is the main class file to generate initial modes.

2. initKmeans.java -- This is the test file that uses KmeansClustering.java to generate initial modes and perform K-means clustering afterwards.

There are three things that needs to be set before executing the initKmeans.java 

- The path of source file name. The file name should be in arff format.

	String targetDir = "//path//to//directory//for//data//"; //Directory Name

	String inputFile =  targetDir+"iris.arff"; // Data file name
	
	The output files for initial centers and data object memberships are written with subscript "-centers.csv" and "-membership.csv" to the input file name.
			
- The number of clusters in the data. It can be done by altering this line 

	initkm.setK(N);//number of clusters

	where N is the number of clusters in the data
  
- The number of nearest neighbours to merge cluster center strings to arrive at K initial clusters.
	
	km1.setNN(1); //for merging clusters, distance of 'q^th' nearest neighbour of a cluster center.
	
	//For practical purposes a value of 1 is good. Increasing this value to large number will result in exception.
	
If eclipse is not used then the following line can be removed from the top
package initKmean;

Dependencies
------------
This package is tested using Weka-3-9-2, Apache Commons Math 3.6.1 and Apache Common Lang 3.3.7. The respective jars can be obtained from the following hyperlinks:

https://www.cs.waikato.ac.nz/ml/weka/downloading.html

http://commons.apache.org/proper/commons-math/download_math.cgi

http://commons.apache.org/proper/commons-lang/download_lang.cgi

Citation
---------
If you use this program in your research and publish a paper, then please use the following citation

@article{khan2004cluster,
  title={Cluster center initialization algorithm for K-means clustering},
  author={Khan, Shehroz S and Ahmad, Amir},
  journal={Pattern recognition letters},
  volume={25},
  number={11},
  pages={1293--1302},
  year={2004},  
  publisher={Elsevier}
}

		
# ccia
