package matcha.utils;

import java.util.Arrays;

import matcha.engine.Tensor;
import matcha.utils.math.LinAlg;

public class Tensors {

	public static String toString(Tensor t){
		try{
			StringBuilder sb = new StringBuilder();
			sb.append("(Tensor of shape ");
			sb.append(t.formatShape());
			sb.append(", gradEnabled=");
			sb.append(t.gradEnabled());
			sb.append(")\n");

			sb.append(toString(t.data(), t.shape(), new int[t.shape().length], t.shape().length-1, new StringBuilder()));

			return sb.toString();
		}
		catch (Exception e) {
			return e.toString();
		}
	}

	private static String toString(double[] data, int[] shape, int[] idxs, int d, StringBuilder sb) throws Exception{
		if (d == 1){
			sb.append("[");
			idxs[1] = -1;
			for (int i = 0; i < shape[0]*shape[1]; i++){
				idxs[1]++;
				int j = 1;
				if(idxs[j] > shape[j] - 1){
					idxs[j--] = 0;
					idxs[j]++;
					sb.append("\n");
					for(int sp = 0; sp < shape.length - d; sp++){
						sb.append(" ");
					}
				}
				sb.append(data[LinAlg.rmo(data.length, shape, idxs)]);
				if (i != shape[0] * shape[1] - 1)
					sb.append(", ");
			}
			sb.append("]");
			return sb.toString();
		}
		else {
			sb.append("[");
			for(int i = 0; i < shape[d]; i++){
				int[] new_idxs = idxs.clone();
				new_idxs[d] = i;
				sb.append(toString(data, shape, new_idxs, d-1, new StringBuilder()));
				if (i != shape[d] - 1){
					sb.append(",\n");
					for(int sp = 0; sp < shape.length - d; sp++){
						sb.append(" ");
					}
				}
			}
			sb.append("]");
			return sb.toString();
		}
	}
}
