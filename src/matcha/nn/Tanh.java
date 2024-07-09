package matcha.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.engine.Value;
import matcha.engine.ValueTanh;

public class Tanh extends Module<Value[]>{

	public Tanh() {

	}

	@Override
	public Value[] forward(final Value[] x) {
		final Value[] out = new Value[x.length];
		for(int i = 0; i < x.length; i++) {
			out[i] = new ValueTanh(x[i]);
		}

		return out;
	}

	@Override
	public List<Value> parameters() {
		return new ArrayList<>();
	}

	@Override
	public String toString() {
		return "Tanh()";
	}
}
