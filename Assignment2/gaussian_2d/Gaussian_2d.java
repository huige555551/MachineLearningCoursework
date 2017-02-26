//gaussian_1d
import java.io.*;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeMap;
import java.lang.Math.*;

class DimensionObject2d {
	private double mean;
	private double variance;
	public ArrayList<Double> values=new ArrayList<Double>();
	public double calculateMean(){
		double sum=0.0;
		for(double d: values){
			sum+=d;
		}
		return sum/(double)values.size();
	}
}
class ClassObject2d{
	public DimensionObject2d[] dimensions;
	private double[] mean;
	private double[] covariance;
	public ClassObject2d(int dimensionsLength){
		dimensions= new DimensionObject2d[dimensionsLength];
		for(int i =0;i<dimensionsLength;i++){
			dimensions[i]=new DimensionObject2d();
		}
		mean= new double[dimensionsLength];
		covariance= new double[4];
	}
	public double calculateCoVariance(int r, int c, double meanRow, double meanCol){
		double covariance = 0.0;
		for(int i=0;i<dimensions[r].values.size();i++){
			covariance+=(dimensions[r].values.get(i)-meanRow)*(dimensions[c].values.get(i)-meanCol);
		}
		return covariance/(dimensions[r].values.size()-1);
	}
	public void displayLines(int classLabel){
		for(int i=0;i<dimensions.length;i++){
			mean[i]=dimensions[i].calculateMean();
		}
		covariance[0]=calculateCoVariance(0,0,mean[0],mean[0]);
		covariance[1]=calculateCoVariance(0,1,mean[0],mean[1]);
		covariance[2]=calculateCoVariance(1,0,mean[1],mean[0]);
		covariance[3]=calculateCoVariance(1,1,mean[1],mean[1]);
		System.out.printf("Class %d, mean = [%.2f, %.2f], sigma = [%.2f, %.2f, %.2f, %.2f]\n", classLabel,mean[0],mean[1],covariance[0],covariance[1],covariance[2],covariance[3]);
	}
}
public class Gaussian_2d {
	public void runStuff(String filePath){
		String line="";
		int dimensions=0;
		Map<Integer,ClassObject2d> classObjects = new HashMap<Integer,ClassObject2d>();
		try
		{
			// Reading from input files
			File f = new File(filePath);
	    BufferedReader br = new BufferedReader(new FileReader(f));
	    while ((line = br.readLine()) != null) {
	      String[] lineVariables = line.trim().split("\\s+");
					dimensions = lineVariables.length - 1;
					int classLabel = Integer.parseInt(lineVariables[lineVariables.length-1]);
					if(!classObjects.containsKey(classLabel))
						classObjects.put(classLabel,new ClassObject2d(2));
					for(int i=0;i<2;i++){
						classObjects.get(classLabel).dimensions[i].values.add(Double.parseDouble(lineVariables[i]));
					}
	    }
			br.close();
		}
		catch(Exception ex){
			System.out.println(ex);
		}
    // Calculating the mean and variance for each classLabel and dimension and displaying them
		for(int key: classObjects.keySet()){
			classObjects.get(key).displayLines(key);
		}
	}
	public static void main(String args[]){
		Gaussian_2d obj = new Gaussian_2d();
		obj.runStuff(args[0]);
	}
}
