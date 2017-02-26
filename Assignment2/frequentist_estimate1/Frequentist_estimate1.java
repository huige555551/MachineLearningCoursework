import java.io.*;
import java.util.Random;

public class Frequentist_estimate1 {
	public static void main(String args[]){
		String s="";
		Random r = new Random();
		double count = 0;
		for(int i=0;i<3100;i++){
			if(r.nextDouble()<=0.1){
				s+="a";
				count++;
			} else
			{
				s+="b";
			}
		}
		//Calculating probability using Frequentist approach
		System.out.printf("p(c = 'a') = %.4f\n",count/3100);
	}
}
