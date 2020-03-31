//This is the main algorithm that uses prominent attributes and/or all attributes if needed
// merge them
//Author: Shehroz S. Khan
//Affiliation: University of Waterloo, Canada
//Date: May'2012
//LICENCE: Read Separate File

//About: This is the main class to generate initial modes and perform K-modes clustering
//       Details of the algorithm are in the paper

package initKmean;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

class KmeanClustering {

	private int K;
	private int NN;
	private int ITR_MAX;
	private int [] count; 
	private double [][] newMeans;
	private int [][] mCluster ;

	//Constructor
	public KmeanClustering () {
	}

	//Setters 
	public void setK(int k) {
		K = k;
	}

	public void setNN(int nN) {
		NN = nN;
	}

	public void setITR_MAX (int itr){
		ITR_MAX=itr;
	}

	//Getters
	public int getK() {
		return K;
	}

	public int getNN() {
		return NN;
	}

	public double [][] getMeans(){
		return newMeans;
	}

	public int [][] getmCluster() {
		return mCluster;
	}

	public int [] getObjectCountInClusters(){
		return count;
	}

	public int getITR_MAX (){
		return ITR_MAX;
	}


	// CCIA - Cluster Center Initialization Algorithm
	public double [][] CCIA (Instances data, String outputFile) throws Exception {
		System.out.println("Total Instances="+data.numInstances());
		int [][] clusterString = new int [data.numInstances()][data.numAttributes()];
		//Find centers corresponding to each attribute
		for(int i=0;i<data.numAttributes();i++) {
			System.out.println("Atr="+i);
			double [] val = new double[data.numInstances()];
			for (int j=0;j<data.numInstances();j++) {
				val[j]=data.instance(j).value(i);
				//System.out.println("j="+j+", val="+val[j]);
			}
			int [] str=clusterNumericAttribute(val,data);
			//for (int j=0;j<str[i].length;j++)
			//System.out.print(str[i][j]+" ");
			//System.out.println();
			int [] membership=generateClusterString(str,data);
			for(int l=0;l<data.numInstances();l++) {
				clusterString[l][i]=membership[l];
			}			
		} //end for each attributes
		String [] cstr = extractClusterStrings(clusterString,data);
		Map<String, Integer> distinctClassStr = findUniqueClusterStrings(cstr);
		double [][] initCenters = findInitialCenters(cstr,distinctClassStr, data);
		return initCenters;
	}

	//Find Initial Centers for Kmeans clustering
	public double [][] findInitialCenters(String [] cstr, Map<String, Integer> distinctClassStr, Instances data) throws Exception {
		double [][] initCenters = new double [distinctClassStr.size()][data.numAttributes()];
		int [] count = new int [distinctClassStr.size()];
		for(int i=0;i<cstr.length;i++) {
			int j=0;
			Iterator it = distinctClassStr.entrySet().iterator();
			while (it.hasNext()) {
				Map.Entry pairs = (Map.Entry)it.next();
				//System.out.println(pairs.getKey() + " = " + pairs.getValue()+" -->"+pairs.getKey().toString().equals(cstr[i]));
				//Store all strings
				//topclusterString[i]=pairs.getKey().toString();
				if(pairs.getKey().toString().equals(cstr[i])) {
					for(int k=0;k<data.numAttributes();k++) 
						initCenters[j][k]+=data.instance(i).value(k);
					count[j]++;
					break;
				}
				j++;
			}
		}
		
		for (int i=0;i<distinctClassStr.size();i++) {
			for (int j=0;j<data.numAttributes();j++) {
				initCenters[i][j]=initCenters[i][j]/count[i];
				//System.out.print(initCenters[i][j]+" ");
			}
			//System.out.println();
		}
		 
		if (distinctClassStr.size()==getK()) 
			return initCenters;
		else
			return initCenters = MergeDBMSDC(initCenters,distinctClassStr,data);
	}

	//Merge DBMSDC algorithm to merge similar centers
	public double [][] MergeDBMSDC(double [][] initCenters, Map<String, Integer> distinctClassStr , Instances data) throws Exception {
		double [][] centers = new double [getK()][data.numAttributes()];		
		int [] B = new int [distinctClassStr.size()];
		for (int i=0;i<distinctClassStr.size();i++)
			B[i]=i;
		int L;
		for(L=0;L<getK()-1;L++) {
			if(B.length <= getNN())
				throw new Exception ("\n***ATTENTION*** The number of nearest neighbours are more than the centers. "
						+ "Consider reducing the number of nearest numbers using the function setNN() in testKmeans.java");
			double [] R = new double [B.length];
			for (int i=0;i<B.length;i++) {
				double [] distance = new double [B.length];
				for (int j=0;j<B.length;j++) {
					EuclideanDistance ed = new EuclideanDistance();					
					distance[j]=ed.compute(initCenters[i], initCenters[j]);				
				}
				double [] sort= Arrays.copyOf(distance, distance.length);
				Arrays.sort(sort);
				R[i]=sort[getNN()];
			}
			DescriptiveStatistics stat = new DescriptiveStatistics(R);
			double minR = stat.getMin();
			int index=0;
			for (int i=0;i<R.length;i++) {
				if(R[i]==minR) {
					index=i;
					break;
				}
			}

			ArrayList<Integer> S = new ArrayList<Integer>();
			for (int i=0;i<B.length;i++) {
				EuclideanDistance ed = new EuclideanDistance();					
				double dist = ed.compute(initCenters[index], initCenters[i]);
				if (dist < 1.5*minR) {
					S.add(B[i]);
					B=ArrayUtils.removeAllOccurences(B, B[i]);
				}
			}
			double [] temp = new double [data.numAttributes()];
			for (int i=0;i<S.size();i++) {
				for(int j=0;j<data.numAttributes();j++){
					temp[j]+=initCenters[S.get(i)][j]/S.size();
				}
			}
			centers[L]=temp;
		}
		//Merge the remaining centers as the final center
		double [] temp = new double [data.numAttributes()];
		for (int i=0;i<B.length;i++) {
			for(int j=0;j<data.numAttributes();j++) {
				temp[j]+=initCenters[B[i]][j]/B.length;
			}
		}
		centers[L]=temp;
//		for(int j=0;j<data.numAttributes();j++)
//			centers[L][j]=initCenters[B[0]][j];
//			System.out.println("Desired number of "+getK()+" clusters are generated");
		return centers;
	}

	//Extract clustering strings for the whole data
	public String [] extractClusterStrings(int [][] clusterString, Instances data) {
		//Convert numeric class string to character strings
		String [] cstr = new String [data.numInstances()];
		for(int i=0;i<data.numInstances();i++) {
			cstr[i]="";
			for(int j=0; j<data.numAttributes()-1;j++){
				cstr[i]+=clusterString[i][j]+",";
				//System.out.print(clusterString[i][j]+", ");
			}
			cstr[i]+=clusterString[i][data.numAttributes()-1];
			//System.out.println(cstr[i]);
			//System.out.println();
		}
		return cstr;
	}

	//Find unique cluster strings
	public Map<String, Integer> findUniqueClusterStrings(String [] cstr) {
		//Find distinct class strings
		Map<String, Integer> distinctClassStr = distinctAttributes(cstr);
		//System.out.println(distinctClassStr);
		System.out.println("\nDistinct Cluster Strings="+distinctClassStr.size()+"\n");
		return distinctClassStr;
	}

	//Sort a map by value
	public LinkedHashMap sortByValue(Map<String, Integer> map) {
		List list = new LinkedList(map.entrySet());
		Collections.sort(list, new Comparator() {
			public int compare(Object o1, Object o2) {
				return ((Comparable) ((Map.Entry) (o1)).getValue())
						.compareTo(((Map.Entry) (o2)).getValue());
			}
		});

		Map result = new LinkedHashMap();
		for (Iterator it = list.iterator(); it.hasNext();) {
			Map.Entry entry = (Map.Entry)it.next();
			result.put(entry.getKey(), entry.getValue());
		}
		return (LinkedHashMap) result;
	} //end sortByValue

	//Generate cluster strings for each attribute
	public int [] generateClusterString(int [] str, Instances data) {
		//Find new centers corresponding to this attributes cluster allotments
		//Allot data objects based on cluster allotments
		double [][] clust = new double [getK()][data.numAttributes()];
		int [] count = new int [getK()];
		for (int i=0;i<str.length;i++) {
			for (int j=0;j<data.numAttributes();j++) {
				clust[str[i]][j]+=data.instance(i).value(j);
			}	
			count[str[i]]++;
		}
		for (int i=0;i<getK();i++) {
			for (int j=0;j<data.numAttributes();j++) {
				clust[i][j]=clust[i][j]/count[i];
			}
		}
		System.out.println("\nUsing centers derived from this attribute");
		//Perform Kmeans with these initial centers
		int [] membership = KMeansClustering(data,clust,getK(),data.numAttributes());
		return membership;
	}

	public void writeOutput (double [][] initCenters, String outputFile) throws IOException {
		BufferedWriter output = new BufferedWriter(new FileWriter(outputFile));
		for(int i=0;i<initCenters.length;i++) {
			for (int j=0;j<initCenters[i].length;j++) {
				output.write(initCenters[i][j]+",");
			}
			output.newLine();
		}
		output.close();
	}

	//Cluster numeric attribute
	public int [] clusterNumericAttribute(double [] attrib,Instances data) {
		double [][] xs = new double [getK()][1];
		// Normalize attribute values
		DescriptiveStatistics stats = new DescriptiveStatistics(attrib);
		double mean = stats.getMean();
		double sd = stats.getStandardDeviation();
		//System.out.println("m="+mean+" sd="+sd);
		for (int i=0; i<getK();i++) {
			double percentile=(double)(2*(i+1)-1)/(2*getK());
			double z = Math.sqrt(2) * Erf.erfcInv(2*percentile); //https://stats.stackexchange.com/questions/71788/percentile-to-z-score-in-php-or-java
			xs[i][0]=z*sd+mean;
			//System.out.println("p="+percentile+ " z="+z+" xs["+i+"]="+xs[i][0]);
		}
		//Convert 'this' attribute to Weka Instances
		Instances ad = new Instances(data,data.numInstances());
		for (int i=0;i<attrib.length;i++) {
			Instance ins = new DenseInstance(1);
			ins.setValue(0, attrib[i]);
			ad.add(i, ins);
		}

		//System.out.println(ad);;
		//Perform Kmeans on 'this' attribute using xs as initial centers
		int [] membership = KMeansClustering(ad,xs,getK(),1);
		return membership;				
	}

	// K-means Clustering
	public int[] KMeansClustering (Instances data, double [][] means, int K,int numAttr){
		int membership [] = new int [data.numInstances()];
		for(int itr=0;itr<ITR_MAX;itr++) {
			count = new int [K];
			mCluster = new int [K][data.numInstances()];

			System.out.println("------------------ITR="+(itr+1)+"-----------------------");
			//Partition data based on initial means
			for(int i=0;i<data.numInstances();i++){
				double distance [] = new double [K];
				for(int j=0;j<K;j++){
					double [] attr = new double [numAttr]; 
					double [] mattr = new double [numAttr]; 
					for(int k=0;k<numAttr;k++) {
						attr[k] = data.instance(i).value(k);
						mattr[k]=means[j][k];
					}
					//distance[j] = computeHammingDistance(str,modes[j], data.numAttributes());
					EuclideanDistance ed = new EuclideanDistance();					
					distance[j]=ed.compute(attr, mattr);

				} //end for j
				//Find membership of instances to clusters w.r.t. hamming distance
				//for(int j=0;j<K;j++)
				//System.out.print(distance[j]+" ");
				membership[i] = findClusterMembership(distance);
				//System.out.println("m="+membership[i]+" c="+count[membership[i]]);
				mCluster[membership[i]][count[membership[i]]]=i;
				count[membership[i]]++;
			} //end for i
			//for(int i=0;i<K;i++)
			//	System.out.print("cluster["+i+"]="+count[i]+" ");
			//System.out.println();
			//Allocate instances to K clusters
			newMeans = new double [K][numAttr];
			for(int i=0;i<K;i++){
				for(int k=0;k<numAttr;k++) {
					double [] val = new double [count[i]];
					for(int j=0;j<count[i];j++){
						//System.out.print(data.instance(mCluster[i][j]).stringValue(k)+" ");
						val[j] = data.instance(mCluster[i][j]).value(k);
					} //end for j
					DescriptiveStatistics stats = new DescriptiveStatistics(val);
					newMeans[i][k]=stats.getMean();					 
					//System.out.print(newMeans[i][k]+" ");					
				} //end for k
				//System.out.println();
			} //end for i
			//System.out.println();

			//Check termination condition
			//System.out.println("D="+data.numAttributes());
			int flag=1;
			for(int i=0;i<K;i++){
				for(int j=0;j<numAttr;j++){
					if(means[i][j]==newMeans[i][j]) flag*=1;
					else flag=0;
				}
			}
			if(flag==0) means=newMeans;
			else if (flag==1) break;

			//Count empty clusters
			int emptyCluster=0;
			for (int i=0;i<K;i++) {
				if (count[i]==0) {
					emptyCluster++;
				}
			}
			//Reduce the number of clusters K
			K=K-emptyCluster;


		} //end for itr


		return membership;

	}

	//Find cluster membership of a data object
	public int findClusterMembership(double[] distance) {
		double [] temp = new double [distance.length];
		for(int i=0;i<temp.length;i++) temp[i]=distance[i];
		Arrays.sort(temp);
		int i;
		for(i=0;i<distance.length;i++){
			if(temp[0]==distance[i])
				break;
		}
		return i;
	}	

	// Compute Distinct Attribute Values
	public Map<String, Integer> distinctAttributes (String [] args){
		Map<String, Integer> m = new HashMap<String, Integer>();
		for (String a : args) {
			Integer freq = m.get(a);
			m.put(a, (freq == null) ? 1 : freq + 1);
		}
		//System.out.println(m.size() + " distinct words:" + m);
		return m;
	}

	//Read input file
	public Instances readInputFile(String inputCSVfile) throws Exception {
		DataSource source = new DataSource(inputCSVfile);
		Instances data = source.getDataSet();
		// setting class attribute if the data format does not provide this information
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		//System.out.println(data);

		return data;
	} //end readInputFile

	//Remove class attribute
	public Instances removeClassAttribute(Instances data) throws Exception {
		// generate data for clusterer (w/o class)
		Remove filter = new Remove();
		filter.setAttributeIndices("" + (data.classIndex() + 1));
		filter.setInputFormat(data);
		Instances dataClusterer = Filter.useFilter(data, filter);

		return dataClusterer;
	} //end for removeClassAttribute()


	//Normalize data
	public Instances normalizeData(Instances data) throws Exception {
		//normalize
		Normalize normalizeFilter = new Normalize();
		normalizeFilter.setInputFormat(data);
		return data = Filter.useFilter(data, normalizeFilter);
	}

}
