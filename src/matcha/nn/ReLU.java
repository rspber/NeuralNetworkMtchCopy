package matcha.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.engine.Value;
import matcha.engine.ValueRelu;

public class ReLU extends Module<Value[]>{

	@Override
	Value[] forward(final Value[] x) {
		final Value[] out = new Value[x.length];
		for(int i = 0; i < x.length; i++) {
			out[i] = new ValueRelu(x[i]);
		}

		return out;
	}

	@Override
	List<Value> parameters() {
		return new ArrayList<>();
	}

	@Override
	public String toString() {
		return "ReLU()";
	}
	
}
