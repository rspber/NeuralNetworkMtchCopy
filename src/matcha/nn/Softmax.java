package matcha.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.engine.Value;
import matcha.engine.ValueDiv;
import matcha.engine.ValueDouble;
import matcha.engine.ValueExp;
import matcha.engine.ValueUtil;

public class Softmax extends Module<Value[]>{

	@Override
	final Value[] forward(final Value[] x) {
		final Value[] out = new Value[x.length];
		double norm = 0.0;
		
		for(int i = 0; i < x.length; i++) {
			out[i] = new ValueExp(x[i]);
			norm += out[i].data();
		}
		for(int i = 0; i < x.length; i++) {
//			out[i] = ValueUtil.div(out[i], norm);
			out[i] = new ValueDiv(out[i], new ValueDouble(norm));
		}

		return out;
	}

	@Override
	List<Value> parameters() {
		return new ArrayList<>();
	}

	@Override
	public String toString() {
		return "Softmax()";
	}
}
