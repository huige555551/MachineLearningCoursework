//gaussian_1d
import java.io.*;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeMap;
import java.lang.Math.*;

class DimensionObject {
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
	public double calculateVariance(double mean){
		double variance = 0.0;
		for(double d: values){
			variance+=Math.pow(d-mean,2);
		}
		return variance/(values.size()-1);
	}
}
class ClassObject{
	public DimensionObject[] dimensions;
	private double[] mean;
	private double[] variance;
	public ClassObject(int dimensionsLength){
		dimensions= new DimensionObject[dimensionsLength];
		for(int i =0;i<dimensionsLength;i++){
			dimensions[i]=new DimensionObject();
		}
		mean= new double[dimensionsLength];
		variance= new double[dimensionsLength];
	}
	void displayLines(int classLabel){
		for(int i=0;i<dimensions.length;i++){
			mean[i]=dimensions[i].calculateMean();
			variance[i]=dimensions[i].calculateVariance(mean[i]);
			System.out.printf("Class %d, dimension %d, mean = %.2f, variance = %.2f\n", classLabel, i+1,mean[i],variance[i]);
		}
	}
}
public class Gaussian_1d {
	public void runStuff(String filePath){
		String line="";
		int dimensions=0;
		Map<Integer,ClassObject> classObjects = new HashMap<Integer,ClassObject>();
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
						classObjects.put(classLabel,new ClassObject(dimensions));
					for(int i=0;i<dimensions;i++){
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
		Gaussian_1d obj = new Gaussian_1d();
		obj.runStuff(args[0]);
	}
}
