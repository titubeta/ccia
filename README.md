This distribution contains two java files:

1. KmeansClustering.java -- This is the main class file to generate initial modes.

2. initKmeans.java -- This is the test file that uses KmeansClustering.java to generate initial modes and perform K-means clustering afterwards.

There are three things that needs to be set before executing the initKmeans.java 

- The path of source file name. The file name should be in arff format.
	String targetDir = "//path//to//directory//for//data//"; //Directory Name
	String inputFile =  targetDir+"iris.arff"; // Data file name
			
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
This package is tested using Weka-3-9-2, Apache Commons Math 3.6.1 and Apache Common Lang 3.3.7. The respective jars can be obtained from
https://www.cs.waikato.ac.nz/ml/weka/downloading.html
http://commons.apache.org/proper/commons-math/download_math.cgi
http://commons.apache.org/proper/commons-lang/download_lang.cgi

		
# ccia
