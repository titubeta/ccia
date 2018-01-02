package initKmean;

import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.core.Instances;

//About: This is the test class to perform cluster center initialization for K-means algorithms
//This implementation is based on the following paper - 
//Cluster Center Initialization Algorithm for K-mean Clustering, Shehroz S. Khan and Amir Ahmad, Pattern Recognition Letters, Volume 25, No. 11, pages 1293-1302, 2004 
//Online available at - https://pdfs.semanticscholar.org/0288/181f90c5f85ba219ebc4beb7c759fd052408.pdf
//Author: Shehroz S. Khan
//Affiliation: University of Toronto, Canada
//Date: Dec'2017
//LICENCE: Read Separate File

public class testKMeans{
	public static void main(String[] args) throws Exception {
		
        //Read input file
		String targetDir = "//home//shehroz//workspace//Clustering//data//";
		String inputFile =  targetDir+"wine.arff";
		String outputFile = inputFile+"-centers.csv";
		String memFile = inputFile+"-membership.csv";
		
		KmeanClustering km1 = new KmeanClustering();
		//Sets the parameters
	    km1.setK(3); //Number of clusters in the data
		km1.setITR_MAX(1000); //Max iterations for K-means to converge
		km1.setNN(1); //for merging clusters, distance of 'q^th' nearest neighbour of a cluster center.
		//For practical purposes a value of 1 is good. Increasing this value to large number will result in exception.
		
		//Read input file
		Instances data = km1.readInputFile(inputFile);
		//Remove class attribute
		Instances clusterdata = km1.removeClassAttribute(data);
		//Normalize data
		clusterdata = km1.normalizeData(clusterdata);
		
		//Perform CCIA
		double [][] initCenters = km1.CCIA(clusterdata,outputFile);
		//Write initial centers to file
		km1.writeOutput(initCenters, outputFile);
		
		//Use these initial centers to perform clustering
		System.out.println("\n***Using the initial centers computed by CCIA to run K-means algorithm");
		int [] membership = km1.KMeansClustering(clusterdata, initCenters, km1.getK(), clusterdata.numAttributes());
		BufferedWriter output = new BufferedWriter(new FileWriter(memFile));
		for(int i=0;i<membership.length;i++) 
			output.write(membership[i]+"\n");
		output.close();
		System.out.println("\n>>>Initial Centers are written in "+outputFile);
		System.out.println(">>>The membership of each data object (after applying K-means with CCIA centers) is written in "+memFile);
		
		} //end of main
} //end of class