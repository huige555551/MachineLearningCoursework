import java.io.*;
import java.util.Random;

public class Frequentist_estimate2 {
	private static int counters[]=new int[5];
	private static Random r;
	public static void updateCounters(double result){
		if(result > 0.11){
			counters[3]++;
			if(result > 0.12) {
				counters[4]++;
			}
		} else if(result < 0.09){
			counters[1]++;
			if(result < 0.08) {
				counters[0]++;
			}
		} else {
			counters[2]++;
		}
	}
	public static void runSimulation(StringBuffer s, double count){
		for(int i=0;i<3100;i++){
			if(r.nextDouble()<=0.1){
				s.append("a");
				count++;
			} else
			{
				s.append("b");
			}
		}
		updateCounters(count/3100);
	}
	public static void main(String args[]){
		r = new Random();
		for(int i=0;i<10000;i++){
			runSimulation(new StringBuffer(""),0);
		}
		// Output
		System.out.printf("In %d of the simulations p(c = 'a') < 0.08.\n",counters[0]);
		System.out.printf("In %d of the simulations p(c = 'a') < 0.09.\n",counters[1]);
		System.out.printf("In %d of the simulations p(c = 'a') is in the interval [0.09, 0.11].\n",counters[2]);
		System.out.printf("In %d of the simulations p(c = 'a') > 0.11.\n",counters[3]);
		System.out.printf("In %d of the simulations p(c = 'a') > 0.12.\n",counters[4]);
	}
}
